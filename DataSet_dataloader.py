import torch
import torchvision


dataset = torchvision.dataset.MNIST(
    root='./data', transform = torchvision.transforms.ToTensor())
