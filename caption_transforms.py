
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
class CaptionTransform():
    """
    A class to apply random text augmentations to captions.

    Args:
        replace_p (float): Probability of replacing a word with a synonym.
        delete_p (float): Probability of deleting a word.
        swap_p (float): Probability of swapping two words.
        shuffle_p (float): Probability of shuffling the order of words.

    """
    def __init__(self, replace_p=0.1, delete_p=0.1, swap_p=0.1, shuffle_p=0.1):
        self.replace_p = replace_p
        self.delete_p = delete_p
        self.swap_p = swap_p
        self.shuffle_p = shuffle_p

    def __call__(self, caption):
        """
        Apply random text augmentations to the given caption.

        Args:
            caption (str): The input caption to augment.

        Returns:
            str: The augmented caption.

        """
        caption = self.replace_synonyms(caption)
        caption = self.delete_words(caption)
        caption = self.swap_words(caption)
        caption = self.shuffle_words(caption)
        return caption

    def replace_synonyms(self, caption):
        """
        Replace some words in the caption with their synonyms.

        Args:
            caption (str): The input caption to replace words.

        Returns:
            str: The caption with some words replaced with synonyms.

        """
        if random.random() < self.replace_p:
            words = caption.split()
            for i in range(len(words)):
                if random.random() < self.replace_p:
                    synonyms = wordnet.synsets(words[i])
                    if synonyms:
                        replacement = synonyms[0].lemmas()[0].name()
                        words[i] = replacement
            caption = " ".join(words)
        return caption

    def delete_words(self, caption):
        """
        Delete some words from the caption.

        Args:
            caption (str): The input caption to delete words.

        Returns:
            str: The caption with some words deleted.

        """
        if random.random() < self.delete_p:
            words = caption.split()
            n = int(self.delete_p * len(words))
            indices = random.sample(range(len(words)), n)
            words = [word for i, word in enumerate(words) if i not in indices]
            caption = " ".join(words)
        return caption

    def swap_words(self, caption):
        """
        Swap some pairs of adjacent words in the caption.

        Args:
            caption (str): The input caption to swap words.

        Returns:
            str: The caption with some pairs of adjacent words swapped.

        """
        if random.random() < self.swap_p:
            words = caption.split()
            n = int(self.swap_p * len(words))
            indices = random.sample(range(len(words)), n)
            for i in range(0, n, 2):
                j = min(i+1, n-1)
                words[indices[i]], words[indices[j]] = words[indices[j]], words[indices[i]]
            caption = " ".join(words)
        return caption


    def shuffle_words(self, caption):
        """
        Shuffle the order of the words in the caption.

        Args:
            caption (str): The input caption to shuffle words.

        Returns:
            str: The caption with the order of words shuffled.

        """
        if random.random() < self.shuffle_p:
            words = caption.split()
            random.shuffle(words)
            caption = " ".join(words)
        return caption
    
class SimCLRData_Caption_Transform():
    """
    Transforms the input text data for SimCLR training by applying various data augmentations.

    Args:
        p (float): Probability of applying each data augmentation. Default: 0.2.
    """

    def __init__(self, p=0.2):
        """
        Initializes the SimCLRData_Caption_Transform.

        Args:
            p (float): Probability of applying each data augmentation. Default: 0.2.
        """
        self.transform = transforms.Compose([
            CaptionTransform(replace_p=p, delete_p=p, swap_p=p, shuffle_p=p)
        ])
        self.seed = random.randint(0, 2**32 - 1)

    def __call__(self, x):
        """
        Applies the SimCLR data augmentation to the input text data.

        Args:
            x (str): Input text data.

        Returns:
            list: A list of two augmented versions of the input text data.
        """
        random.seed(self.seed)
        return [self.transform(x), self.transform(x)]

