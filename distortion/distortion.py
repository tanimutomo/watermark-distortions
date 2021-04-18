"""
# Watermark Distortions

## List of Distortions
- Identity
- Dropout
- Cropout
- Crop
- Resize
- Gaussian Blur
- JPEG Compression
"""

import abc
import math
import random
import typing

import kornia
import torch
import torch.nn.functional as F


MEAN: typing.List[float] = [0.0, 0.0, 0.0]
STD:  typing.List[float] = [1.0, 1.0, 1.0]


def init(mean: typing.List[float], std: typing.List[float]):
    global MEAN, STD
    MEAN = mean
    STD = std


class Distortioner(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.tensor(MEAN)[None, :, None, None], requires_grad=False)
        self.std =  torch.nn.Parameter(torch.tensor(STD)[None, :, None, None], requires_grad=False)

    def forward(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        return self._normalize(self.distort(self._unnormalize(i_co), self._unnormalize(i_en)))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class Identity(Distortioner):
    def __init__(self):
        super().__init__()

    def distort(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        return i_en


class Dropout(Distortioner):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def distort(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.p == 1: return i_en
        mask = self._create_mask(i_co.shape)
        mask = mask.to(i_co.device)
        return i_en * mask + i_co * (1 - mask)
    
    def _create_mask(self, shape: tuple) -> torch.FloatTensor:
        return torch.where(torch.rand(shape) > self.p,
                           torch.ones(shape), torch.zeros(shape))


class Cropout(Distortioner):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def distort(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.p == 1: return i_en
        mask = self._create_mask(i_co.shape)
        mask = mask.to(i_co.device)
        return i_en * mask + i_co * (1 - mask)
    
    def _create_mask(self, shape: tuple) -> torch.FloatTensor:
        _, _, h, w = shape
        area = h * w
        crop_area = area * self.p

        ch = int(max([random.random() * h, crop_area / w + 1]))
        cw = int(max([crop_area / ch, 1]))
        h_center_range = (ch//2 + 1, h - (ch//2 + 1))
        w_center_range = (cw//2 + 1, w - (cw//2 + 1))
        h_center = random.randint(*h_center_range)
        w_center = random.randint(*w_center_range)

        mask = torch.zeros(shape)
        mask[..., h_center-ch//2:h_center+ch//2, w_center-cw//2:w_center+cw//2] = 1
        return mask


class Crop(Distortioner):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def distort(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.p == 1: return i_en
        h, w, cy, cx = self._sample_params(i_en.shape)
        return i_en[..., cy-h//2 : cy+h//2, cx-w//2 : cx+w//2]
    
    def _sample_params(self, shape: tuple) -> typing.Tuple[int, int, int, int]:
        _, _, h, w = shape
        crop_area = h * w * self.p
        mu = math.pow(crop_area, 0.5)
        sigma = mu / 4
        ch_ = round(random.gauss(mu, sigma))
        ch = max(min(ch_, h-2),  math.ceil(crop_area / h)+2)
        cw = round(crop_area / ch)

        center_range = ((ch//2 + 1, h - (ch//2 + 1)), (cw//2 + 1, w - (cw//2 + 1)))
        center = (random.randint(*center_range[0]), random.randint(*center_range[1]))

        return ch, cw, center[0], center[1]


class Resize(Distortioner):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def distort(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.p == 1: return i_en
        return F.interpolate(i_en, scale_factor=self.p, recompute_scale_factor=True)


class GaussianBlur(Distortioner):
    def __init__(self, w: int, s: float):
        super().__init__()
        self.w = w
        self.s = s
    
    def distort(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.s == 0: return i_en
        return kornia.filters.gaussian_blur2d(i_en, (self.w, self.w), (self.s, self.s))


class JPEGBase(Distortioner):
    def __init__(self):
        super().__init__()
        self.dct_kernel = torch.nn.Parameter(self._create_dct_basis_matrix(), requires_grad=False)
        self.idct_kernel = torch.nn.Parameter(self._create_idct_basis_matrix(), requires_grad=False)

    def distort(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        _, _, h, w = i_en.shape
        if not (h % 8 == 0 and w % 8 == 0): raise TypeError("x shape cannot be devided 8")
        x = kornia.color.rgb_to_yuv(i_en) * 255
        x = self._apply_dct(x)
        x = self.compress(x)
        x = self._apply_idct(x)
        x = kornia.color.yuv_to_rgb(x) / 255
        return x

    def _apply_dct(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self._convdot(self._convdot(x, self.dct_kernel, False), self.idct_kernel, True)

    def _apply_idct(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self._convdot(self._convdot(x, self.idct_kernel, False), self.dct_kernel, True)

    def _convdot(self, x, k: torch.FloatTensor, dot_x_k: bool) -> torch.FloatTensor:
        ys = []
        for i in range(8):
            stride = (1, 8) if dot_x_k else (8, 1)
            pad = (7-i, i, 7, 7) if dot_x_k else (7, 7, 7-i, i)

            kernel = self._get_conv_kernel_1(k, i, dot_x_k, x.device)
            y = F.conv2d(F.pad(x, pad), kernel, stride=stride, groups=3)

            kernel = self._get_conv_kernel_2(x.device)
            y = F.conv_transpose2d(y, kernel, stride=stride, padding=(7, 7), groups=3)

            pad = (i, 7-i, 0, 0) if dot_x_k else (0, 0, i, 7-i)
            y = F.pad(y, pad)

            ys.append(y)

        y = torch.stack(ys, dim=0)
        return torch.sum(y, 0)

    def _create_dct_basis_matrix(self, n=8) -> torch.FloatTensor:
        mat = torch.zeros((n, n))
        for j in range(n):
            mat[0, j] = math.sqrt(2/n) / math.sqrt(2)
        for i in range(1, n):
            for j in range(n):
                mat[i, j] = math.sqrt(2/n) \
                           * math.cos((math.pi/n) * i * (j + 0.5))
        return mat

    def _create_idct_basis_matrix(self) -> torch.FloatTensor:
        return self._create_dct_basis_matrix(8).transpose(1, 0)

    def _get_conv_kernel_1(self, k: torch.Tensor, i: int, dot_x_k: bool, device: torch.device) -> torch.Tensor:
        basis = k[:, i] if dot_x_k else k[i, :]
        pos = (7, slice(7-i, 15-i)) if dot_x_k else (slice(7-i, 15-i), 7)
        kernel = torch.zeros(15, 15, device=device)
        kernel[pos[0], pos[1]] = basis
        return self._expand(kernel)

    def _get_conv_kernel_2(self, device: torch.device) -> torch.Tensor:
        kernel = torch.zeros(15, 15, device=device)
        kernel[7, 7] = 1
        return self._expand(kernel)

    def _expand(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x[None, None, ...].expand(3, 1, -1, -1)

    def compress(self, x: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


class JPEGCompression(JPEGBase):
    def __init__(self, qf: int):
        super().__init__()
        self.qf = qf
        self.kernel = torch.nn.Parameter(self._create_quantization_table(), requires_grad=False)

    def compress(self, x: torch.FloatTensor) -> torch.FloatTensor:
        _, _, h, w = x.shape
        qt = self.kernel.repeat(1, h//8, w//8)
        return torch.round(x.div(qt)) * qt

    def _create_quantization_table(self) -> torch.FloatTensor:
        base_qt_y = torch.tensor(quantization_table_y, dtype=torch.float32).view(8, 8)
        base_qt_uv = torch.tensor(quantization_table_uv, dtype=torch.float32).view(8, 8)
        qt_y = self._adjust_quantization_quality(base_qt_y)
        qt_uv = self._adjust_quantization_quality(base_qt_uv)
        return torch.stack([qt_y, qt_uv, qt_uv], dim=0)
        
    def _adjust_quantization_quality(self, base: torch.FloatTensor) -> torch.FloatTensor:
        s = 5000 / self.qf if self.qf < 50 else 200 - 2*self.qf
        qt = torch.floor((s * base + 50) / 100)
        return torch.where(qt == 0, torch.ones_like(qt), qt)


class JPEGDifferential(JPEGBase):
    def __init__(self):
        super().__init__()
    
    def compress(self, x: torch.FloatTensor) -> torch.FloatTensor:
        _, _, h, w = x.shape
        return self._normalize(self._unnormalize(x) * self.kernel.repeat(1, h//8, w//8))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class JPEGMask(JPEGDifferential):
    def __init__(self):
        super().__init__()
        self.kernel = torch.nn.Parameter(self._create_mask_kernel(), requires_grad=False)
    
    def _create_mask_kernel(self):
        kernel = torch.zeros(3, 8, 8)
        kernel[0, :5, :5] = 1 # y
        kernel[1:, :3, :3] = 1 # u, v
        return kernel


class JPEGDrop(JPEGDifferential):
    def __init__(self):
        super().__init__()
        self.kernel = torch.nn.Parameter(self._create_drop_kernel(), requires_grad=False)
    
    def _create_drop_kernel(self):
        qt_y = torch.tensor(quantization_table_y, dtype=torch.float64).view(8, 8)
        qt_uv = torch.tensor(quantization_table_uv, dtype=torch.float64).view(8, 8)
        prob_y = qt_y.sub(qt_y.min()).div(qt_y.max() - qt_y.min() + 1.0)
        prob_uv = qt_uv.sub(qt_uv.min()).div(qt_uv.max() - qt_uv.min() + 1.0)

        one = torch.ones(8, 8)
        zero = torch.zeros(8, 8)
        kernel_y = torch.where(torch.rand(8, 8) > prob_y, one, zero)
        kernel_u = torch.where(torch.rand(8, 8) > prob_uv, one, zero)
        kernel_v = torch.where(torch.rand(8, 8) > prob_uv, one, zero)

        return torch.stack([kernel_y, kernel_u, kernel_v])


quantization_table_y = [
    16, 11, 10, 16, 24,  40,  51,  61,
    12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,
    14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,
    24, 36, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]
quantization_table_uv = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]
