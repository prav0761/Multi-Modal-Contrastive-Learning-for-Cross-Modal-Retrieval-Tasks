
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
            transform = transforms.ToTensor()
            return transform(image), caption

class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        token_file_path: str,
        caption_index_1: int = 0,
        caption_index_2: int = 1,
        image_transform=None,
        evaluate=False
    ):
        """
        Dataset for Flickr30k images with corresponding captions.

        Parameters:
        root_dir (str): path to the root directory of the dataset.
        token_file_path (str): path to the file containing the captions for the images.
        caption_index_1 (int): index of the first caption to use (default is 0).
        caption_index_2 (int): index of the second caption to use (default is 1).
        image_transform (callable): function to apply to the image(s) (default is None).
        """
        self.root_dir = root_dir
        self.token_file_path = token_file_path
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor()
               
            ]
        )
        self.captions = self._load_captions()
        self.caption_index_1 = caption_index_1
        self.caption_index_2 = caption_index_2
        self.image_transform = image_transform
        if evaluate:
            self.evaluate=evaluate
    def _load_captions(self) -> dict:
        """
        Loads the captions from the token file and returns a dictionary where
        the keys are the image filenames and the values are the corresponding
        captions.

        Returns:
        dict: dictionary with image filenames as keys and lists of captions as values.
        """
        with open(self.token_file_path) as tokenfile:
            captions = tokenfile.readlines()
        caption_dict = {}
        for caption in captions:
            caption_parts = caption.strip().split("#")
            image_file_name = caption_parts[0]
            caption_text_parts = caption_parts[1].split("\t")
            caption_number = int(caption_text_parts[0].replace("#", ""))
            caption_text = caption_text_parts[1]
            if image_file_name not in caption_dict:
                caption_dict[image_file_name] = []
            caption_dict[image_file_name].append(caption_text)
        return caption_dict

    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.

        Returns:
        int: number of images in the dataset.
        """
        return len(self.captions)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns the image and its corresponding captions at the given index.

        Parameters:
        idx (int): index of the image to retrieve.

        Returns:
        tuple: tuple containing the two images and two corresponding captions.
        """
        image_filename = list(self.captions.keys())[idx]
        image_path = os.path.join(self.root_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        captions = self.captions[image_filename]
        if self.image_transform:
            # Apply image transforms to create two transformed images
            img1, img2 = self.image_transform(image)
            return (
                self.transform(image),
                img1,
                img2,
                captions[self.caption_index_1],
                captions[self.caption_index_2],
            )
        else:
            return (
                self.transform(image),
                captions[0],
                captions[1],
                captions[2],
                captions[3],
                captions[4]
            )
