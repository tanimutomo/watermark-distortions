import typing

import matplotlib.pyplot as plt
import PIL
import ptdt
import torch
import torchvision
import torchvision.transforms.functional as F

import distortion


MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.436, 0.615]
MEAN_TENSOR = torch.tensor(MEAN)[None, :, None, None]
STD_TENSOR = torch.tensor(STD)[None, :, None, None]


def main():
    inp = get_image_tensor("./images/input.jpg")
    inp = normalize(inp)

    torch.set_printoptions(precision=4, sci_mode=False)

    distortioners = {
        "identity": distortion.Identity(MEAN, STD),
        "dropout": distortion.Dropout(0.5, MEAN, STD),
        "cropout": distortion.Cropout(0.5, MEAN, STD),
        "crop": distortion.Crop(0.5, MEAN, STD),
        "resize": distortion.Resize(0.5, MEAN, STD),
        "gaussian_blur": distortion.GaussianBlur(5, 2, MEAN, STD),
        "jpeg_compression": distortion.JPEGCompression(50, MEAN, STD),
        "jpeg_mask": distortion.JPEGMask(MEAN, STD),
        "jpeg_drop": distortion.JPEGDrop(MEAN, STD)
    }

    outs = []
    for name, dis in distortioners.items():
        print(name)
        out = dis(inp, inp)
        outs.append(padding_image(unnormalize(out), inp.shape))

    imsave(torch.cat(outs, dim=0), "./images/distortion.jpg", nrow=3, pad=2)
    imshow(F.to_tensor(PIL.Image.open("./images/distortion.jpg")))


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


def normalize(x: torch.Tensor):
    return (x - MEAN_TENSOR) / STD_TENSOR
    

def unnormalize(x: torch.Tensor):
    return x * STD_TENSOR + MEAN_TENSOR
    

def get_image_tensor(path: str) -> torch.FloatTensor:
    img = PIL.Image.open(path)
    img = adjust_image_size(img)
    return F.to_tensor(img)[None, ...]


def padding_image(x: torch.FloatTensor, shape: tuple) -> torch.FloatTensor:
    if x.shape == shape: return x
    _, _, h, w = x.shape
    out = torch.zeros(shape)
    out[:, :, :h, :w] = x
    return out


if __name__ == "__main__":
    main()