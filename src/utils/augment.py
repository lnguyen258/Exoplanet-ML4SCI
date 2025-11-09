from typing import *
import random

import torch
from torch import nn, Tensor
import torchvision.transforms as T


class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if random.random() < self.p:
            x = self.fn(x)
        return x

def augment_compose(image_size: int) -> nn.Module:
    return T.Compose([
        RandomApply(T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), p=0.8),
        RandomApply(T.Grayscale(num_output_channels=3), p=0.2),
        RandomApply(T.RandomHorizontalFlip(p=0.5), p=0.8),
        RandomApply(T.GaussianBlur(kernel_size=(3, 3), sigma=(1.5, 1.5)), p=0.2),
        T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
        T.ToTensor(),
        T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    ])