
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import transforms
from PIL import Image
import requests
import io
from io import BytesIO
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import json
import threading
import torchvision.transforms.functional as F
import torchvision.models as models
import torch.nn as nn


def get_gpu_stats():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print('No of GPUs i have is',torch.cuda.device_count())
    print(torch.cuda.current_device())
    print('My Graphic Card is',torch.cuda.get_device_name(torch.cuda.current_device()))
    print('Is Cuda Available',torch.cuda.is_available())
def layerwise_trainable_parameters(model):
    total_trainable_params = 0
    for name, param in model.named_parameters():
        num_trainable_params = 0
        if param.requires_grad:
            num_trainable_params = param.numel()
            total_trainable_params += num_trainable_params
        print(f'Layer name: {name}\tNumber of trainable parameters: {num_trainable_params}')
    print(f'Total number of trainable parameters: {total_trainable_params}')
def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

