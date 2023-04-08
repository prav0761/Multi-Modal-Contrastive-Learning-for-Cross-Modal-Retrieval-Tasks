
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import transforms
from PIL import Image
import requests
import time
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
from torch.nn.utils.rnn import pad_sequence
import random
from tqdm import tqdm
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
import threading
import torchvision.models as models
import torch.nn as nn


# In[3]:


def train(dataloader, image_model, text_model, optimizer_image, optimizer_text, criterion,device,
          scheduler_image=None, scheduler_text=None, trade_off_ii=1, trade_off_cc=1,trade_off_ic=1,trade_off_ci=1):
    """
    Trains the image and text models using the provided dataloader and optimizer.

    Parameters:
        dataloader (torch.utils.data.DataLoader): The dataloader used for training.
        image_model (torch.nn.Module): The image model to be trained.
        text_model (torch.nn.Module): The text model to be trained.
        optimizer_image (torch.optim.Optimizer): The optimizer used for training the image model.
        optimizer_text (torch.optim.Optimizer): The optimizer used for training the text model.
        criterion (torch.nn.Module): The loss function used for training.
        scheduler_image (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler for the image optimizer.
        scheduler_text (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler for the text optimizer.
        trade_off_ili (float, optional): The trade off between image-image loss and text-text loss. Defaults to 1.
        trade_off_cc (float, optional): The trade off between caption-caption loss and image-caption loss. Defaults to 1.

    Returns:
        float: The average loss over the epoch.
    """
    loss_epoch = 0

    for idx, batch in enumerate(dataloader):
        image_model.train()
        text_model.train()

        batch_size = batch[0].shape[0]
        image1, image2, caption1, caption2 = batch[0], batch[1], batch[3], batch[4]

        _, embed_image1 = image_model(image1, device)
        _, embed_image2 = image_model(image2, device)
        _, embed_caption1 = text_model(caption1, device)
        _, embed_caption2 = text_model(caption2, device)

        contrastive_loss = (trade_off_ii * criterion(embed_image1, embed_image2, batch_size) +
                      trade_off_cc * criterion(embed_caption1, embed_caption2, batch_size) +
                      trade_off_ic * criterion(embed_image1, embed_caption2, batch_size) +
                     trade_off_ci * criterion(embed_caption1, embed_image2, batch_size) )

        contrastive_loss.backward()

        optimizer_image.step()
        optimizer_text.step()

        optimizer_image.zero_grad()
        optimizer_text.zero_grad()
        
        loss_epoch += contrastive_loss.item()

        del batch, image1, image2, caption1, caption2, embed_image1, embed_image2, embed_caption1, embed_caption2, contrastive_loss
        torch.cuda.empty_cache()
    if scheduler_image:
        scheduler_image.step()
    if scheduler_text:
        scheduler_text.step()
    epoch_loss = loss_epoch / len(dataloader)
    return epoch_loss
def test(dataloader, image_model, text_model, criterion, device, trade_off_ii=1, trade_off_cc=1,trade_off_ic=1,trade_off_ci=1):
    """
    Calculate the loss of the model using dataloader, image model, text model,
    and criterion on the given device with the given trade_off values.

    Args:
    dataloader (DataLoader): The dataloader to use for iterating over the data.
    image_model (nn.Module): The image model used to extract image features.
    text_model (nn.Module): The text model used to extract text features.
    criterion (nn.Module): The criterion used to calculate the loss.
    device (str): The device to use for computation.
    trade_off_ii (float, optional): The trade off value for image features. Default is 1.
    trade_off_cc (float, optional): The trade off value for text features. Default is 1.

    Returns:
    float: The epoch loss.
    """

    loss_epoch = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            image_model.eval()
            text_model.eval()
            batch_size = batch[0].shape[0]
            image1, image2, caption1, caption2 = batch[0], batch[1], batch[3], batch[4]

            _, embed_image1 = image_model(image1, device)
            _, embed_image2 = image_model(image2, device)
            _, embed_caption1 = text_model(caption1, device)
            _, embed_caption2 = text_model(caption2, device)

            contrastive_loss = (trade_off_ii * criterion(embed_image1, embed_image2, batch_size) +
                      trade_off_cc * criterion(embed_caption1, embed_caption2, batch_size) +
                      trade_off_ic * criterion(embed_image1, embed_caption2, batch_size) +
                     trade_off_ci * criterion(embed_caption1, embed_image2, batch_size) )

            loss_epoch += contrastive_loss.item()

            del batch, image1, image2, caption1, caption2, embed_image1, embed_image2, embed_caption1, embed_caption2, contrastive_loss
            torch.cuda.empty_cache()

    epoch_loss = loss_epoch / len(dataloader)
    return epoch_loss


# In[5]:



