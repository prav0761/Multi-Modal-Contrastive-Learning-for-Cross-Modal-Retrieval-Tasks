
# coding: utf-8

# In[1]:

import torch
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
import requests
import time
import numpy as np
import io
from io import BytesIO
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence
import random
from tqdm import tqdm
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
import threading
import torchvision.models as models
import torch.nn as nn
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel
from nltk.corpus import wordnet
from caption_transforms import SimCLRData_Caption_Transform
from image_transforms import SimCLRData_image_Transform
from dataset import FlickrDataset,Flickr30kDataset
from models import ResNetSimCLR,OpenAI_SIMCLR,Image_fine_tune_model ,Text_fine_tune_model
from metrics import inter_ContrastiveLoss, intra_ContrastiveLoss,cosine_sim , finetune_ContrastiveLoss
from metrics import LARS,Optimizer_simclr
from logger import Logger
from train_fns import train, test , fine_tune_train ,fine_tune_val

def get_gpu_stats():
    """
    Prints the GPU stats including device type, number of GPUs, current device, graphic card name and availability.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device type: {device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Graphic card name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")

def layerwise_trainable_parameters(model):
    """
    Prints the number of trainable parameters in each layer of the model and the total number of trainable parameters.
    """
    total_trainable_params = 0
    for name, param in model.named_parameters():
        num_trainable_params = 0
        if param.requires_grad:
            num_trainable_params = param.numel()
            total_trainable_params += num_trainable_params
        print(f"Layer name: {name}\tNumber of trainable parameters: {num_trainable_params}")
    print(f"Total number of trainable parameters: {total_trainable_params}")

def count_trainable_parameters(model) -> int:
    """
    Returns the total number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def get_gpu_memory():
    """
    Prints the amount of free GPU memory in gigabytes.
    """
    torch.cuda.empty_cache()
    mem_allocated = torch.cuda.memory_allocated()
    mem_cached = torch.cuda.memory_reserved()
    mem_free = torch.cuda.get_device_properties(0).total_memory - mem_allocated - mem_cached
    mem_free_gb = mem_free / 1e9
    print(f"Free GPU memory: {mem_free_gb:.2f} GB")
    
    

def recall_score_calculate(image_embed, text_embeds, top_k, image_to_txt=True):
    """
    Calculates the recall score between an image embedding and a list of text embeddings.

    Args:
        image_embed (torch.Tensor): The image embedding.
        text_embeds (List[torch.Tensor]): The list of text embeddings.
        top_k (int): The number of top indices to retrieve.
        image_to_txt (bool): If True, calculates the cosine similarity between the image and text embeddings.

    Returns:
        The recall score.
    """
    topk_indices_list = []
    for text_embed in text_embeds:
        # compute cosine similarity between image and text embeddings
        if image_to_txt:
            similarities = cosine_sim(image_embed, text_embed)
        else:
            similarities = cosine_sim(text_embed, image_embed)
        topk_indices = torch.topk(similarities, k=top_k, dim=1)[1]
        topk_indices_list.append(topk_indices)
    recalls=[]
    topk_indices_new=torch.concat(topk_indices_list[:],dim=1)
    for i, indices in enumerate(topk_indices_new):
        if i in indices:
            recalls.append(1)
        else:
            recalls.append(0)
    recall_score = sum(recalls) / len(recalls)
    return recall_score


def recall_score_calculate_travel(image_embed, text_embed, top_k, image_to_txt=True):
    """
    Calculates the recall score between an image embedding and a single text embedding.

    Args:
        image_embed (torch.Tensor): The image embedding.
        text_embed (torch.Tensor): The text embedding.
        top_k (int): The number of top indices to retrieve.
        image_to_txt (bool): If True, calculates the cosine similarity between the image and text embeddings.

    Returns:
        The recall score.
    """
    if image_to_txt:
        similarities = cosine_sim(image_embed, text_embed)
    else:
        similarities = cosine_sim(text_embed, image_embed)

    topk_indices = torch.topk(similarities, k=top_k, dim=1)[1]
    recalls = []
    for i, indices in enumerate(topk_indices):
        if i in indices:
            recalls.append(1)
        else:
            recalls.append(0)

    recall_score = sum(recalls) / len(recalls)
    return recall_score
def get_all_recall_scores(image_embed, text_embeds):
    """
    Returns recall scores for images to text (it) and text to images (ti)
    with top_k values of 1, 5, and 10.
    
    Parameters:
        image_embed (array): Embedding for the image.
        text_embeds (array): List of embeddings for each text.
    
    Returns:
        tuple: A tuple containing the recall scores for 
        top_k = 1, 5, and 10 for both it and ti.
    """
    r_1_it = recall_score_calculate(image_embed, text_embeds, top_k=1, image_to_txt=True)
    r_5_it = recall_score_calculate(image_embed, text_embeds, top_k=5, image_to_txt=True)
    r_10_it = recall_score_calculate(image_embed, text_embeds, top_k=10, image_to_txt=True)
    r_1_ti = recall_score_calculate(image_embed, text_embeds, top_k=1, image_to_txt=False)
    r_5_ti = recall_score_calculate(image_embed, text_embeds, top_k=5, image_to_txt=False)
    r_10_ti = recall_score_calculate(image_embed, text_embeds, top_k=10, image_to_txt=False)
    
    return r_1_it, r_5_it, r_10_it, r_1_ti, r_5_ti, r_10_ti

def get_img_txt_embed(images, txt1, txt2, txt3, txt4, txt5, image_model, text_model, device):
    """
    Given a set of images and texts, returns their embeddings using image and text models.

    Args:
    images: List of images.
    txt1: First text.
    txt2: Second text.
    txt3: Third text.
    txt4: Fourth text.
    txt5: Fifth text.
    image_model: Image model used for embedding.
    text_model: Text model used for embedding.
    device: Device used for processing.

    Returns:
    Tuple of image embeddings and list of text embeddings.
    """
    image_model.eval()
    text_model.eval()
    
    # Compute image embeddings.
    image_embed = image_model(torch.stack(images), device, single=False)
    
    # Compute text embeddings.
    text_embeds = []
    text_embeds.append(text_model(txt1, device, single=False))
    text_embeds.append(text_model(txt2, device, single=False))
    text_embeds.append(text_model(txt3, device, single=False))
    text_embeds.append(text_model(txt4, device, single=False))
    text_embeds.append(text_model(txt5, device, single=False))
    
    return image_embed, text_embeds
