
# coding: utf-8

# In[1]:

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
#from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import json
import threading
import torchvision.transforms.functional as F
from nltk.corpus import wordnet
import torchvision.transforms.functional as F
def get_color_distortion(s=1.0):
    """
    Applies color distortion to an image.

    Args:
        s (float): Strength of color distortion.

    Returns:
        A color distortion transform.
    """
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.2),
        rnd_gray,
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)], p=0.8)])
    return color_distort

class RandomGaussianBlur(object):
    """
    Applies random Gaussian blur to an image.

    Args:
        p (float): Probability of applying Gaussian blur.
        min_sigma (float): Minimum value of sigma for Gaussian blur.
        max_sigma (float): Maximum value of sigma for Gaussian blur.

    Returns:
        A random Gaussian blur transform.
    """
    def __init__(self, p=0.5, min_sigma=0.1, max_sigma=2.0):
        self.p = p
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.min_sigma, self.max_sigma)
            kernel_size = int(0.1 * min(img.size))
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            kernel_size = max(1, kernel_size)
            img = F.gaussian_blur(img, kernel_size, [sigma, sigma])
        return img


class SimCLRData_image_Transform():
    """
    Applies a set of image transformations used in SimCLR.

    Args:
        size (int): Size of the output image.

    Returns:
        A SimCLR image transformation object.
    """
    def __init__(self, size=224):
        s = 1
        size=224
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(),
            get_color_distortion(s=1.0),
            RandomGaussianBlur(p=0.5, min_sigma=0.1, max_sigma=2.0),
            transforms.ToTensor()
        ])
        self.seed = random.randint(0, 2**32 - 1)

    def __call__(self, x):
        """
        Applies the set of image transformations to an input image.

        Args:
            x (PIL.Image.Image): Input image.

        Returns:
            A list of two transformed images.
        """
        random.seed(self.seed)
        return [self.transform(x), self.transform(x)]

