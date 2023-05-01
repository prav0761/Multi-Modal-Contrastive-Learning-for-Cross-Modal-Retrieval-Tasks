
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


def train(dataloader,data_type, image_model, text_model, optimizer_image, optimizer_text, intra_criterion,inter_criterion,device,
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
        intra_contrastive_loss=0
        batch_size = batch[0].shape[0]
        if data_type=='flickr_travel':
            image1, image2, caption1, caption2 = batch[0], batch[1], batch[3], batch[4]
        if data_type=='flickr30k':
            image1, image2, caption1, caption2 = batch[1], batch[2], batch[3], batch[4]
            
        intra_image,inter_image = image_model(image1, device)
        intra_image1,inter_image1 = image_model(image2, device)
        intra_contrastive_loss+=(trade_off_ii * intra_criterion(intra_image, intra_image1, batch_size))
        del intra_image , intra_image1
        intra_cap,inter_cap = text_model(caption1, device)
        intra_cap1,inter_cap1 = text_model(caption2, device)
        intra_contrastive_loss+=(trade_off_cc * intra_criterion(intra_cap, intra_cap1, batch_size))

            
            
        ci_loss, ic_loss=inter_criterion(inter_image,inter_image1,inter_cap,inter_cap1)
        del  inter_image,inter_image1,inter_cap,inter_cap1
        inter_contrastive_loss= trade_off_ci*ci_loss + trade_off_ic*ic_loss
        total_loss = intra_contrastive_loss + inter_contrastive_loss
        total_loss.backward()
        optimizer_image.step()
        optimizer_text.step()

        optimizer_image.zero_grad()
        optimizer_text.zero_grad()
        loss_epoch += total_loss.item()
    if scheduler_image:
        scheduler_image.step()
    if scheduler_text:
        scheduler_text.step()
    epoch_loss = loss_epoch / len(dataloader)
    return epoch_loss
def test(dataloader, data_type, image_model, text_model,intra_criterion,inter_criterion, device, trade_off_ii=1,
         trade_off_cc=1,trade_off_ic=1,trade_off_ci=1):
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
            if data_type=='flickr_travel':
                image1, image2, caption1, caption2 = batch[0], batch[1], batch[3], batch[4]
            if data_type=='flickr30k':
                image1, image2, caption1, caption2 = batch[1], batch[2], batch[3], batch[4]
            

            intra_image,inter_image = image_model(image1, device)
            intra_image1,inter_image1 = image_model(image2, device)
            intra_cap,inter_cap = text_model(caption1, device)
            intra_cap1,inter_cap1 = text_model(caption2, device)

            intra_contrastive_loss = (trade_off_ii * intra_criterion(intra_image, intra_image1, batch_size) +
                                trade_off_cc * intra_criterion(intra_cap, intra_cap1, batch_size))



            
            
            ci_loss, ic_loss=inter_criterion(inter_image,inter_image1,inter_cap,inter_cap1)
            del  inter_image,inter_image1,inter_cap,inter_cap1
            inter_contrastive_loss= trade_off_ci*ci_loss + trade_off_ic*ic_loss
            total_loss = intra_contrastive_loss + inter_contrastive_loss

            loss_epoch += total_loss.item()



    epoch_loss = loss_epoch / len(dataloader)
    return epoch_loss


# In[5]:

def fine_tune_train(data_loader, image_model, text_model, data_type, device, criterion,
                    optimizer_image, optimizer_text, scheduler_image=None, scheduler_text=None, caption_idx=None):
    """
    Trains the image and text models on the given data_loader.

    Parameters:
    data_loader: DataLoader
        The data loader that contains the image and caption data.
    image_model: nn.Module
        The image model to be fine-tuned.
    text_model: nn.Module
        The text model to be fine-tuned.
    data_type: str
        The type of data loader being used.
    device: str
        The device to train on.
    criterion: nn.Module
        The loss function.
    optimizer_image: torch.optim.Optimizer
        The optimizer for the image model.
    optimizer_text: torch.optim.Optimizer
        The optimizer for the text model.
    scheduler_image: torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler for the image model.
    scheduler_text: torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler for the text model.
    caption_idx: int
        The index of the caption data in the data loader.

    Returns:
    float: The epoch loss.
    """
    image_model.train()
    text_model.train()
    loss_epoch = 0
    for idx, batch in enumerate(data_loader):
        if data_type == 'flickr_travel':
            image, caption = batch[0], batch[1]
        if data_type == 'flickr30k':
            image, caption = batch[0], batch[caption_idx]

        img_embed = image_model(image, device, single=False)
        cap_embed = text_model(caption, device, single=False)

        loss = criterion(img_embed, cap_embed, batch[0].shape[0]) + criterion(cap_embed, img_embed, batch[0].shape[0])
        # loss = 0.1 * criterion(img_embed, cap_embed) 
        loss.backward()
        optimizer_image.step()
        optimizer_text.step()

        optimizer_image.zero_grad()
        optimizer_text.zero_grad()
        loss_epoch += loss.item()
    epoch_loss = loss_epoch / len(data_loader)
    if scheduler_image:
        scheduler_image.step()
    if scheduler_text:
        scheduler_text.step()

    return round(epoch_loss, 4)


def fine_tune_val(data_loader, image_model,text_model,data_type,device,criterion, caption_idx=None):
    """
    Validates the image and text models on the given data_loader.

    Parameters:
    data_loader: DataLoader
        The data loader that contains the image and caption data.
    image_model: nn.Module
        The image model to be validated.
    text_model: nn.Module
        The text model to be validated.
    data_type: str
        The type of data loader being used.
    device: str
        The device to validate on.
    criterion: nn.Module
        The loss function.
    caption_idx: int
        The index of the caption data in the data loader.

    Returns:
    float: The epoch loss.
    """
    loss_epoch=0
    for idx, batch in enumerate(data_loader):
        image_model.eval()
        text_model.eval()
        if data_type=='flickr_travel':
            image , caption = batch[0], batch[1]
        if data_type=='flickr30k':
            image,caption = batch[0], batch[caption_idx]

        img_embed = image_model(image,device,single=False)
        cap_embed = text_model(caption,device,single=False)


        loss=criterion(img_embed,cap_embed,batch[0].shape[0]) + criterion(cap_embed,img_embed,batch[0].shape[0])
        #loss=0.1*criterion(img_embed,cap_embed)
        loss_epoch += loss.item()
    epoch_loss = loss_epoch / len(data_loader)
    return round(epoch_loss,4)

