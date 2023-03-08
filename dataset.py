
# coding: utf-8

# In[ ]:

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
class FlickrDataset(Dataset):
    def __init__(self, rootdir, data_dir, dataset_type, image_transform=None, caption_transform=None):
        """
        Args:
            rootdir (str): Root directory containing caption files.
            data_dir (str): Directory containing image files.
            dataset_type (str): 'train' or 'test'.
            image_transform (callable, optional): Optional transform to be applied to the image.
            caption_transform (callable, optional): Optional transform to be applied to the caption.
        """
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.root_dir = rootdir
        self.data_dir = data_dir
        self.image_files = sorted(os.listdir(os.path.join(data_dir)))
        if dataset_type == 'train':
            self.caption_file = os.path.join(rootdir, "train_captions.txt")
        elif dataset_type == 'test':
            self.caption_file = os.path.join(rootdir, "test_captions.txt")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        tag = ""
        caption = ""
        with open(self.caption_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts[0] == self.image_files[idx]:
                    caption = parts[1]
                    break
        image = Image.open(image_path).convert('RGB')
        
        if self.image_transform:
            # Apply image transforms to create two transformed images
            img1, img2 = self.image_transform(image)
            
        if self.caption_transform:
            # Apply caption transforms to create two transformed captions
            caption1, caption2 = self.caption_transform(caption)
            
        # Return appropriate data based on which transforms are applied
        if self.image_transform and not self.caption_transform:
            # If only image transforms are applied, return two transformed images and the original caption
            return img1, img2, caption
        
        elif not self.image_transform and self.caption_transform:
            # If only caption transforms are applied, return the original image and two transformed captions
            return image, caption1, caption2
        
        elif self.image_transform and self.caption_transform:
            # If both image and caption transforms are applied, return two transformed images and two transformed captions
            return img1, img2, caption,caption1, caption2
        
        else:
            # If no transforms are applied, return the original image and caption
            return image, caption

