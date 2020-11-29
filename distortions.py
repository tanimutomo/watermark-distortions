import abc
import math
import random
import typing

import kornia
import torch
import torch.nn.functional as F
import torchvision


class Distortioner(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError()


class Identity(Distortioner):
    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        return i_en


class Dropout(Distortioner):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.p == 1: return i_en
        mask = self._create_mask(i_co.shape)
        return i_en * mask + i_co * (1 - mask)
    
    def _create_mask(self, shape: tuple) -> torch.FloatTensor:
        return torch.where(torch.rand(shape) > self.p,
                           torch.ones(shape), torch.zeros(shape))


class Cropout(Distortioner):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.p == 1: return i_en
        mask = self._create_mask(i_co.shape)
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
        self.p = p

    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.p == 1: return i_en
        h, w, hc, wc = self._sample_params(i_en.shape)
        return i_en[..., hc-h//2 : hc+h//2, wc-w//2 : wc+w//2]
    
    def _sample_params(self, shape: tuple) -> typing.Tuple[int, int, int, int]:
        _, _, h, w = shape
        ch, cw = int(h*self.p), int(w*self.p)

        h_center_range = (ch//2 + 1, h - (ch//2 + 1))
        w_center_range = (cw//2 + 1, w - (cw//2 + 1))
        h_center = random.randint(*h_center_range)
        w_center = random.randint(*w_center_range)

        return ch, cw, h_center, w_center


class Resize(Distortioner):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.p == 1: return i_en
        return F.interpolate(i_en, scale_factor=self.p, recompute_scale_factor=True)


class GaussianBlur(Distortioner):
    def __init__(self, w, s: int):
        self.w = w
        self.s = s
    
    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.s == 0: return i_en
        return kornia.filters.gaussian_blur2d(i_en, (self.w, self.w), (self.s, self.s))


class JPEGCompression(Distortioner):
    def __init__(self):
        self.dct_kernel = self._create_dct_basis_matrix()
        self.idct_kernel = self._create_idct_basis_matrix()
        self.compression_kernel = self._create_compression_kernel()

    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        _, _, h, w = i_en.shape
        if not (h % 8 == 0 and w % 8 == 0): raise TypeError("x shape cannot be devided 8")
        x = kornia.color.rgb_to_yuv(i_en)
        x = self._apply_dct(x)
        x = self._compress(x, self.compression_kernel)
        x = self._apply_idct(x)
        x = kornia.color.yuv_to_rgb(x)
        return x

    def _apply_dct(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self._convdot(self._convdot(x, self.dct_kernel, False), self.idct_kernel, True)

    def _apply_idct(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self._convdot(self._convdot(x, self.idct_kernel, False), self.dct_kernel, True)

    def _convdot(self, x, k: torch.FloatTensor, dot_x_k: bool) -> torch.FloatTensor:
        ys = []
        for i in range(8):
            basis = k[:, i] if dot_x_k else k[i, :]
            kernel = torch.zeros(15, 15)
            pos = (7, slice(7-i, 15-i)) if dot_x_k else (slice(7-i, 15-i), 7)
            kernel[pos[0], pos[1]] = basis

            stride = (1, 8) if dot_x_k else (8, 1)
            pad = (7-i, i, 7, 7) if dot_x_k else (7, 7, 7-i, i)
            y = F.conv2d(F.pad(x, pad), self._expand(kernel), stride=stride, groups=3)

            kernel = torch.zeros(15, 15)
            kernel[7, 7] = 1
            y = F.conv_transpose2d(y, self._expand(kernel), stride=stride, padding=(7, 7), groups=3)

            pad = (i, 7-i, 0, 0) if dot_x_k else (0, 0, i, 7-i)
            y = F.pad(y, pad)

            ys.append(y)

        y = torch.stack(ys, dim=0)
        return torch.sum(y, 0)

    def _compress(self, x, k: torch.FloatTensor) -> torch.FloatTensor:
        _, _, h, w = x.shape
        _, kh, kw = k.shape
        if not (kh == 8 and kw == 8): raise TypeError("k shape is not 8x8")
        return x * k.repeat(1, h//8, w//8)

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

    def _expand(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x[None, None, ...].expand(3, 1, -1, -1)

    def _create_compression_kernel(self) -> torch.FloatTensor:
        raise NotImplementedError()


class JPEGMask(JPEGCompression):
    def __init__(self):
        super().__init__()
    
    def _create_compression_kernel(self):
        kernel = torch.zeros(3, 8, 8)
        kernel[0, :5, :5] = 1 # y
        kernel[1:, :3, :3] = 1 # u, v
        return kernel


class JPEGDrop(JPEGCompression):
    def __init__(self):
        # quantization tables
        qt_y = torch.tensor([
            16, 11, 10, 16, 24,  40,  51,  61,
            12, 12, 14, 19, 26,  58,  60,  55,
            14, 13, 16, 24, 40,  57,  69,  56,
            14, 17, 22, 29, 51,  87,  80,  62,
            18, 22, 37, 56, 68,  109, 103, 77,
            24, 36, 55, 64, 81,  104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99,
        ], dtype=torch.float64).view(8, 8)
        qt_uv = torch.tensor([
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
        ], dtype=torch.float64).view(8, 8)
        self.prob_y = qt_y.sub(qt_y.min()).div(qt_y.max() - qt_y.min() + 1.0)
        self.prob_uv = qt_uv.sub(qt_uv.min()).div(qt_uv.max() - qt_uv.min() + 1.0)
        super().__init__()
    
    def _create_compression_kernel(self):
        one = torch.ones(8, 8)
        zero = torch.zeros(8, 8)
        kernel_y = torch.where(torch.rand(8, 8) > self.prob_y, one, zero)
        kernel_u = torch.where(torch.rand(8, 8) > self.prob_uv, one, zero)
        kernel_v = torch.where(torch.rand(8, 8) > self.prob_uv, one, zero)
        return torch.stack([kernel_y, kernel_u, kernel_v])

