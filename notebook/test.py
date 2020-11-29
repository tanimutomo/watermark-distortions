# # Watermark Distortions
# 
# ## List of Distortions
# - Identity
# - Dropout
# - Cropout
# - Crop
# - Resize
# - Gaussian Blur
# - JPEG Compression

import sys
import typing

sys.path.append("..")

import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
import torchvision.transforms.functional as F

import distortions


def main():
    inp = get_image_tensor("../images/dog_03.jpg")

    torch.set_printoptions(precision=4, sci_mode=False)

    distortioners = {
        "identity": distortions.Identity(),
        "dropout": distortions.Dropout(0.5),
        "cropout": distortions.Cropout(0.5),
        "crop": distortions.Crop(0.5),
        "resize": distortions.Resize(0.5),
        "gaussian_blur": distortions.GaussianBlur(5, 2),
        "jpeg_mask": distortions.JPEGMask(),
        "jpeg_drop": distortions.JPEGDrop()
    }

    outs = []
    for name, dis in distortioners.items():
        print(name)
        out = dis(inp, inp)
        outs.append(padding_image(out, inp.shape))

    imsave(torch.cat(outs, dim=0), "../images/distortions.jpg", nrow=4, pad=2)
    imshow(F.to_tensor(PIL.Image.open("../images/distortions.jpg")))


def imshow(x: torch.FloatTensor):
    x = x[0] if len(x.shape) == 4 else x
    x = x.permute(1, 2, 0)
    plt.imshow(x)


def imsave(x: typing.List[torch.FloatTensor], path: str, nrow, pad: int):
    torchvision.utils.save_image(x, path, nrow=nrow, padding=pad)


def adjust_image_size(img :PIL.Image.Image) -> PIL.Image.Image:
    w, h = img.size
    nh, nw = h - h % 8, w - w % 8
    return F.resize(img, (nh, nw))


def get_image_tensor(path: str) -> torch.FloatTensor:
    img = PIL.Image.open(path)
    img = adjust_image_size(img)kk
    return F.to_tensor(img)[None, ...]


def padding_image(x: torch.FloatTensor, shape: tuple) -> torch.FloatTensor:
    if x.shape == shape: return x
    _, _, h, w = x.shape
    out = torch.zeros(shape)
    out[:, :, :h, :w] = x
    return out


if __name__ == "__main__":
    main()