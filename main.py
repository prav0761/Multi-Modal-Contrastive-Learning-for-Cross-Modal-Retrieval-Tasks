
# coding: utf-8

# In[4]:


import torch
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
from models import ResNetSimCLR,OpenAI_SIMCLR
from utils import get_gpu_stats,layerwise_trainable_parameters,count_trainable_parameters,get_gpu_memory
from metrics import inter_ContrastiveLoss, intra_ContrastiveLoss
from metrics import LARS,Optimizer_simclr
from logger import Logger
from train_fns import train, test
from args import args_c
torch.cuda.empty_cache()


def main(args):
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize trial number
    trial_number = args.trial_number
    total_epochs=args.total_epochs
    intra_projection_dim=args.intra_projection_dim
    inter_projection_dim =args.inter_projection_dim
    image_learning_rate = args.image_learning_rate
    text_learning_rate=args.text_learning_rate
    momentum = args.momentum
    temperature = args.temperature
    weight_decay = args.weight_decay
    optimizer_type = args.optimizer_type
    trade_off_ii=args.trade_off_ii
    trade_off_cc=args.trade_off_cc
    trade_off_ic=args.trade_off_ic
    trade_off_ci=args.trade_off_ci
    batch_size = args.batch_size
    margin= args.margin
    max_violation = args.max_violation
    image_layers_to_train=args.image_layers_to_train
    text_layers_to_train=args.text_layers_to_train
    caption_index_1=args.caption_index_1
    caption_index_2=args.caption_index_2
    scheduler_status=args.scheduler_status
    flickr30k_images_dir_path=args.flickr30k_images_dir_path
    flickr30k_tokens_dir_path=args.flickr30k_tokens_dir_path
    graph_save_dir = args.graph_save_dir
    logresults_save_dir_path=args.logresults_save_dir_path
    train_log = os.path.join(logresults_save_dir_path, f'train{trial_number}_30k.log')
    image_model_log = os.path.join(logresults_save_dir_path, f'image_model{trial_number}_30k.pth')
    text_model_log = os.path.join(logresults_save_dir_path, f'text_model{trial_number}_30k.pth')
    graph_save_dir = graph_save_dir
    print(image_layers_to_train,text_layers_to_train)
# Create train and test datasets using FlickrDataset
    dataset = Flickr30kDataset(flickr30k_images_dir_path, 
                               flickr30k_tokens_dir_path,
                               caption_index_1=caption_index_1,
                               caption_index_2=caption_index_2,
                              image_transform=SimCLRData_image_Transform())
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [29783, 1000, 1000])

    train_loader = DataLoader(train_set, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             num_workers=4, 
                             pin_memory=True)
    val_loader = DataLoader(val_set, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=4, 
                             pin_memory=True)



    # Initialize ResNetSimCLR model
    resnet_model = ResNetSimCLR(
        model='resnet50',
        intra_projection_dim=intra_projection_dim,
        inter_projection_dim =inter_projection_dim,
        layers_to_train=image_layers_to_train,
        evaluate=False
    ).to(device)
    gpt_model = OpenAI_SIMCLR(
        model='openai-gpt',
        intra_projection_dim=intra_projection_dim,
        inter_projection_dim=inter_projection_dim,
        layers_to_train=text_layers_to_train,
        evaluate=False
    ).to(device)

    # Define loss function
    intra_loss=intra_ContrastiveLoss(device,temperature=temperature)
    newinter_loss=inter_ContrastiveLoss(margin=margin, max_violation=max_violation)
    # Define optimizers and schedulers
    optimizer_image = Optimizer_simclr(optimizer_name=optimizer_type,
                                       model_parameters=resnet_model.parameters(),
                                       lr=image_learning_rate,
                                       momentum=momentum,
                                       weight_decay=weight_decay)

    scheduler_image = optimizer_image.scheduler
    optimizer_image = optimizer_image.optimizer

    optimizer_text = Optimizer_simclr(optimizer_name=optimizer_type,
                                      model_parameters=gpt_model.parameters(),
                                      lr=text_learning_rate,
                                      momentum=momentum,
                                      weight_decay=weight_decay)

    scheduler_text = optimizer_text.scheduler
    optimizer_text = optimizer_text.optimizer
    logger_save = Logger(train_log,
                     image_model_log,
                     text_model_log, 
                     optimizer_type, 
                     image_learning_rate,
                     text_learning_rate,
                     weight_decay,
                     batch_size,
                     momentum, 
                     temperature, 
                     total_epochs,
                     trade_off_cc,
                     trade_off_ii,
                     trade_off_ic,
                     trade_off_ci,
                     image_layers_to_train,
                     text_layers_to_train,
                     intra_projection_dim,
                     inter_projection_dim,
                    scheduler=scheduler_status,
                    margin=margin,
                    max_violation=max_violation)
    logger_save.start_training()
    print('started_training')
    # Loop through epochs and train the models
    for epoch in tqdm(range(total_epochs)):

        start = time.time()

        # Train the models and get the loss
        train_loss = train(dataloader=train_loader, 
                               data_type='flickr30k',
                               image_model=resnet_model, 
                               text_model=gpt_model,
                               optimizer_image=optimizer_image, 
                               optimizer_text=optimizer_text, 
                               intra_criterion=intra_loss,
                               inter_criterion=newinter_loss,
                                device=device,
                               scheduler_image=scheduler_image,
                               scheduler_text=scheduler_text,
                               trade_off_ii=trade_off_ii, 
                               trade_off_cc=trade_off_cc,
                               trade_off_ic=trade_off_ic,
                               trade_off_ci=trade_off_ci)

        # Test the models and get the loss
        test_loss = test(dataloader=val_loader, 
                         data_type='flickr30k',
                         image_model=resnet_model,
                         text_model=gpt_model,
                         intra_criterion=intra_loss,
                         inter_criterion=newinter_loss,
                         device=device,
                         trade_off_ii=trade_off_ii,
                         trade_off_cc=trade_off_cc,
                         trade_off_ic=trade_off_ic,
                         trade_off_ci=trade_off_ci)

        end = time.time()

        # Log the results of the epoch
        logger_save.log(epoch + 1, resnet_model, gpt_model, train_loss, test_loss, end - start)

    # End training and plot the losses
    logger_save.end_training()
    print('training_end')
    logger_save.plot_losses(trial_number,
                            graph_save_dir,
                            optimizer_type, 
                            image_learning_rate,
                            text_learning_rate,
                            weight_decay, 
                            batch_size, 
                            momentum, 
                            temperature, 
                           total_epochs,
                            trade_off_cc,
                            trade_off_ii,
                            trade_off_ci,
                            trade_off_ic,
                            image_layers_to_train,
                            text_layers_to_train,
                            intra_projection_dim,
                          inter_projection_dim,
                          scheduler=scheduler_status)
    
if __name__ == '__main__':
    # Parse command-line arguments
    args = args_c()

    # Call the main function with the parsed arguments
    main(args)

