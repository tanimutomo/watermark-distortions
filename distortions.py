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
        mask = self._create_mask(i_co.shape())
        return i_en * mask + i_co * (1 - mask)
    
    def _create_mask(self, shape: tuple) -> torch.FloatTensor:
        torch.where(torch.rand(shape) > self.p, 1, 0)


class Cropout(Distortioner):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.p == 1: return i_en
        mask = self._create_mask(i_co.shape())
        return i_en * mask + i_co * (1 - mask)
    
    def _create_mask(self, shape: tuple) -> torch.FloatTensor:
        _, _, h, w = shape
        area = h * w
        crop_area = area * self.p

        ch = max([random.random() * min([crop_area, h]), 1])
        cw = max([area / ch, 1])
        h_center_range = (ch//2 + 1, h - (ch//2 + 1))
        w_center_range = (cw//2 + 1, h - (cw//2 + 1))
        h_center = random.randint(*h_center_range)
        w_center = random.randint(*w_center_range)

        mask = torch.zeros(shape)
        mask[..., h_center-ch:h_center+ch, w_center-cw:w_center+cw] = 1
        return mask


class Crop(Distortioner):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.p == 1: return i_en
        h, w, hc, wc = self._sample_params(i_co.shape)
        return i_en[..., hc-h : hc+h, wc-w : wc+w]
    
    def _sample_params(self, shape: tuple) -> typing.Tuple[int, int, int, int]:
        _, _, h, w = shape
        area = h * w
        crop_area = area * self.p

        ch = max([random.random() * min([crop_area, h]), 1])
        cw = max([area / ch, 1])
        h_center_range = (ch//2 + 1, h - (ch//2 + 1))
        w_center_range = (cw//2 + 1, h - (cw//2 + 1))
        h_center = random.randint(*h_center_range)
        w_center = random.randint(*w_center_range)

        return ch, cw, h_center, w_center


class Resize(Distortioner):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.p == 1: return i_en
        return F.interpolate(i_en, self.p)


class GaussianBlur(Distortioner):
    def __init__(self, s: int):
        self.s = s
    
    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        if self.s == 0: return i_en
        return kornia.filters.gaussian_blur2d(i_en, (self.p, self.p), (1, 1))


class JPEGCompression(Distortioner):
    def __init__(self):
        self.dct_kernel = self._mat_to_kernel(self._create_dct_basis_matrix())
        self.idct_kernel = self._mat_to_kernel(self._create_idct_basis_matrix())
        self.compression_kernel = self._create_compression_kernel()

    def __call__(self, i_co, i_en: torch.FloatTensor) -> torch.FloatTensor:
        x = kornia.color.rgb_to_yuv(i_en)
        x = F.conv2d(x, self.dct_kernel, stride=8)
        x = F.conv2d(x, self.compression_kernel, stride=8)
        x = F.conv2d(x, self.idct_kernel, stride=8)
        x = kornia.color.yuv_to_rgb(x)
        return x

    def _create_dct_basis_matrix(self, n=8) -> torch.FloatTensor:
        mat = torch.zeros((n, n))
        for j in range(n):
            mat[0, j] = math.sqrt(2/n) / math.sqrt(2)
        for i in range(1, n):
            for j in range(n):
                mat[i, j] = math.sqrt(2/n) \
                           * math.cos((math.pi/n) * i * (j + 0.5))

    def _create_idct_basis_matrix(self, n=8) -> torch.FloatTensor:
        return self._create_dct_basis_matrix(n).transpose()

    def _mat_to_kernel(self, mat: torch.FloatTensor) -> torch.FloatTensor:
        return mat[None, None, ...].expand(3, 3, -1, -1)

    def _create_compression_kernel(self) -> torch.FloatTensor:
        raise NotImplementedError()


class JPEGMask(JPEGCompression):
    def __init__(self):
        super().__init__()
    
    def _create_compression_kernel(self):
        kernel = torch.zeros(3, 3, 8, 8)
        kernel[:, 0, :5, :5] = 1 # y
        kernel[:, 1:, :3, :3] = 1 # u, v
        return kernel


class JPEGDrop(JPEGCompression):
    def __init__(self):
        super().__init__()
        quantization_table_y = torch.tensor([
            16, 11, 10, 16, 24,  40,  51,  61,
            12, 12, 14, 19, 26,  58,  60,  55,
            14, 13, 16, 24, 40,  57,  69,  56,
            14, 17, 22, 29, 51,  87,  80,  62,
            18, 22, 37, 56, 68,  109, 103, 77,
            24, 36, 55, 64, 81,  104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99,
        ]).view(8, 8)
        quantization_table_uv = torch.tensor([
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
        ]).view(8, 8)
        self.prob_y = quantization_table_y / (quantization_table_y.max()+1)
        self.prob_uv = quantization_table_uv / 100
    
    def _create_compression_kernel(self):
        kernel_y = torch.where(torch.rand(8, 8) > self.prob_y, 1, 0)
        kernel_u = torch.where(torch.rand(8, 8) > self.prob_uv, 1, 0)
        kernel_v = torch.where(torch.rand(8, 8) > self.prob_uv, 1, 0)
        return torch.stack([kernel_y, kernel_u, kernel_v])[None, ...].expand(3, -1, -1, -1)
