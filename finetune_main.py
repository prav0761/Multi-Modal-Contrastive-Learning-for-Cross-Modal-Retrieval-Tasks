
# coding: utf-8

# In[5]:


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
from utils import layerwise_trainable_parameters,count_trainable_parameters,get_gpu_memory,recall_score_calculate
from utils import get_all_recall_scores,get_img_txt_embed
from metrics import inter_ContrastiveLoss, intra_ContrastiveLoss,cosine_sim , finetune_ContrastiveLoss
from metrics import LARS,Optimizer_simclr
from logger import Logger ,Fine_Tune_Logger
from train_fns import train, test , fine_tune_train ,fine_tune_val
from args import args_c , args_finetune
torch.cuda.empty_cache()
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    """
    This function contains the main code.
    """
    trial_number = args.trial_number
    total_epochs=args.total_epochs
    data_type=args.data_type
    output_dim =args.output_dim
    image_learning_rate = args.image_learning_rate
    text_learning_rate=args.text_learning_rate
    momentum = args.momentum
    temperature = args.temperature
    weight_decay = args.weight_decay
    optimizer_name = args.optimizer_type
    batch_size = args.batch_size
    caption_idx_eval=args.caption_idx_eval
    scheduler_status=args.scheduler_status
    flickr30k_images_dir_path=args.flickr30k_images_dir_path
    flickr30k_tokens_dir_path=args.flickr30k_tokens_dir_path
    logresults_save_dir_path=args.logresults_save_dir_path
    train_log = os.path.join(logresults_save_dir_path, f'finetune{trial_number}_30k.log')
    image_model_log = os.path.join(logresults_save_dir_path, f'image_model_finetune{trial_number}_30k.pth')
    text_model_log = os.path.join(logresults_save_dir_path, f'text_model_finetune{trial_number}_30k.pth')
    image_model_weights_file=args.image_weights_file
    text_model_weights_file=args.text_weights_file

    dataset = Flickr30kDataset(flickr30k_images_dir_path, 
                                   flickr30k_tokens_dir_path,
                                   caption_index_1=0,
                                   caption_index_2=1,
                                  image_transform=None,
                                      evaluate=True)
    indices = list(range(len(dataset)))
    train_indices = indices[:29783]
    val_indices = indices[29783:30783]
    test_indices = indices[30783:]
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    train_loader = DataLoader(train_set, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             num_workers=4, 
                             pin_memory=True)
    val_loader = DataLoader(val_set, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             num_workers=4, 
                             pin_memory=True)
    test_loader = DataLoader(test_set, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             num_workers=4, 
                             pin_memory=True)


    images, txt1, txt2, txt3, txt4, txt5, index1 = zip(*[(val_set[i][0], val_set[i][1], val_set[i][2],
                                                          val_set[i][3], val_set[i][4], val_set[i][5], torch.tensor(i)) 
                                                         for i in range(len(val_set))])


    model_finetune_img=Image_fine_tune_model(weights_file=image_model_weights_file,
                                            output_dim=output_dim).to(device)
    model_finetune_text=Text_fine_tune_model(weights_file=text_model_weights_file,
                                            output_dim=output_dim).to(device)


    optimizer_image = Optimizer_simclr(optimizer_name=optimizer_name,
                                       model_parameters=model_finetune_img.parameters(),
                                       lr=image_learning_rate,
                                       momentum=momentum,
                                       weight_decay=weight_decay)

    cont_loss=intra_ContrastiveLoss(device,temperature=temperature)

    #scheduler_image = optimizer_image.scheduler
    optimizer_image = optimizer_image.optimizer

    optimizer_text = Optimizer_simclr(optimizer_name=optimizer_name,
                                       model_parameters=model_finetune_text.parameters(),
                                       lr=text_learning_rate,
                                       momentum=momentum,
                                       weight_decay=weight_decay)

    #scheduler_text = optimizer_text.scheduler
    optimizer_text = optimizer_text.optimizer

    
    scheduler_image = torch.optim.lr_scheduler.MultiStepLR(optimizer_image, milestones=[20], gamma=0.1)
    scheduler_text = torch.optim.lr_scheduler.MultiStepLR(optimizer_text, milestones=[20], gamma=0.1)
    
    
    logger_save = Fine_Tune_Logger(train_log,
                         image_model_log,
                         text_model_log, 
                         optimizer_name, 
                         image_learning_rate,
                         text_learning_rate,
                         weight_decay,
                         batch_size,
                         momentum, 
                         temperature, 
                         total_epochs,
                         output_dim,
                        scheduler=scheduler_status)
    logger_save.start_training()

    for epoch in tqdm(range(total_epochs)):


            start = time.time()
            # Train the models and get the loss
            train_loss = fine_tune_train(data_loader=train_loader, 
                                   image_model=model_finetune_img, 
                                   text_model=model_finetune_text,
                                   data_type=data_type,
                                   device=device,
                                   criterion=cont_loss,
                                   optimizer_image=optimizer_image, 
                                   optimizer_text=optimizer_text, 
                                   scheduler_image=scheduler_image,
                                   scheduler_text=scheduler_text,
                                   caption_idx=caption_idx_eval)


            # Test the models and get the loss
            test_loss = fine_tune_val(data_loader=val_loader, 
                                   image_model=model_finetune_img, 
                                   text_model=model_finetune_text,
                                   data_type=data_type,
                                   device=device,
                                   criterion=cont_loss,
                                   caption_idx=caption_idx_eval)


            image_embed ,text_embeds=get_img_txt_embed(images,txt1,txt2,txt3,txt4,
                                                  txt5,model_finetune_img,model_finetune_text,device)
            r_1_it,r_5_it,r_10_it,r_1_ti,r_5_ti,r_10_ti=get_all_recall_scores(image_embed,text_embeds)
            print('epoch {:03d}: train_loss = {:.4f}, val_loss = {:.4f}, recall@1 = {:.4f}, recall@5 = {:.4f}, recall@10 = {:.4f}'
              .format(epoch, train_loss, test_loss, r_1_it, r_5_it, r_10_it))
            end = time.time()
            logger_save.fine_tune_log(epoch + 1, model_finetune_img, model_finetune_text, r_5_it, end - start)
    logger_save.end_training()

if __name__ == '__main__':
    # Parse command-line arguments
    args = args_finetune()

    # Call the main function with the parsed arguments
    main(args)

