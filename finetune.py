
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
from models import ResNetSimCLR,OpenAI_SIMCLR
from utils import get_gpu_stats,layerwise_trainable_parameters,count_trainable_parameters,get_gpu_memory
from metrics import inter_ContrastiveLoss, intra_ContrastiveLoss,cosine_sim , finetune_ContrastiveLoss
from metrics import LARS,Optimizer_simclr
from logger import Logger
from train_fns import train, test
from args import args_c
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[6]:


flickr30k_images_dir_path='/work/08629/pradhakr/maverick2/cv_project/flickr30k-images'
flickr30k_tokens_dir_path='/work/08629/pradhakr/maverick2/cv_project/flickr30k_captions/results_20130124.token'
caption_index_1=0
caption_index_2=1

dataset = Flickr30kDataset(flickr30k_images_dir_path, 
                               flickr30k_tokens_dir_path,
                               caption_index_1=caption_index_1,
                               caption_index_2=caption_index_2,
                              image_transform=None,
                                  evaluate=True)
indices = list(range(len(dataset)))
train_indices = indices[:29783]
val_indices = indices[29783:30783]
test_indices = indices[30783:]
train_set = torch.utils.data.Subset(dataset, train_indices)
val_set = torch.utils.data.Subset(dataset, val_indices)
test_set = torch.utils.data.Subset(dataset, test_indices)
batch_size=128
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
test_loader = DataLoader(train_set, 
                         batch_size=batch_size, 
                         shuffle=True, 
                         num_workers=4, 
                         pin_memory=True)


# In[ ]:


images=[]
index1=[]
txt1,txt2,txt3,txt4,txt5=[],[],[],[],[]
for i in range(len(test_set)):
    images.append(test_set[i][0])
    txt1.append(test_set[i][1])
    txt2.append(test_set[i][2])
    txt3.append(test_set[i][3])
    txt4.append(test_set[i][4])
    txt5.append(test_set[i][5])
    index1.append(torch.tensor(i))


# In[ ]:



# In[7]:


class Image_fine_tune_model(nn.Module):
    def __init__(self, weights_file,output_dim=1024):
        super(Image_fine_tune_model, self).__init__()
        self.model_resnet = ResNetSimCLR(
            model='resnet50',
            intra_projection_dim=128,
            inter_projection_dim=1024,
            layers_to_train=[],
            evaluate=False
        )
        self.model_resnet.load_state_dict(torch.load(weights_file))
        self.finetune_backbone = self.model_resnet.backbone
        self.fc_layer = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)
        )

    def forward(self, img,device,single=False):
        if single:
            features = self.finetune_backbone(img.to(device).unsqueeze(0))
        else:
            features = self.finetune_backbone(img.to(device))
        features = features.view(features.size(0), -1)
        image_embed = self.fc_layer(features)
        return image_embed
class text_fine_tune_model(nn.Module):
    def __init__(self, weights_file,output_dim=1024):
        super(text_fine_tune_model, self).__init__()
        self.gpt_model = OpenAI_SIMCLR(
                        model='openai-gpt',
                        intra_projection_dim=128,
                        inter_projection_dim=1024,
                        layers_to_train=[],
                        evaluate=True
                    ).to(device)

        self.gpt_model.load_state_dict(torch.load(weights_file))
        self.fc_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, output_dim)
        )

    def forward(self, text,device,single=False):
        if single:
            text_features=self.gpt_model([text],device)
        else:
            text_features=self.gpt_model(text,device)
        text_features = text_features.view(text_features.size(0), -1)
        text_embed = self.fc_layer(text_features)
        return text_embed


# In[8]:


model_finetime_img=Image_fine_tune_model(weights_file='/work/08629/pradhakr/maverick2/cv_project/flickr30k_imagecaption/image_model6_30k.pth',
                                        output_dim=1024).to(device)
model_finetime_text=text_fine_tune_model(weights_file='/work/08629/pradhakr/maverick2/cv_project/flickr30k_imagecaption/text_model6_30k.pth',
                                        output_dim=1024).to(device)


# In[ ]:


optimizer_image = Optimizer_simclr(optimizer_name='adam',
                                   model_parameters=model_finetime_img.parameters(),
                                   lr=0.001,
                                   momentum=0.9,
                                   weight_decay=0.0001)

cont_loss=intra_ContrastiveLoss(device,temperature=0.07)

scheduler_image = optimizer_image.scheduler
optimizer_image = optimizer_image.optimizer

optimizer_text = Optimizer_simclr(optimizer_name='adam',
                                  model_parameters=model_finetime_text.parameters(),
                                  lr=0.001,
                                  momentum=0.9,
                                  weight_decay=0.0001)

scheduler_text = optimizer_text.scheduler
optimizer_text = optimizer_text.optimizer


# In[ ]:


#vse=finetune_ContrastiveLoss(margin=0.2 , max_violation=True)


# In[ ]:


def fine_tune_train(data_loader, imagemodel,text_model,data_type,device,criterion,
                    optimizer_image, optimizer_text, scheduler_image=None,scheduler_text=None, caption_idx=None):
    imagemodel.train()
    text_model.train()
    loss_epoch=0
    for idx, batch in enumerate(data_loader):
        imagemodel.train()
        text_model.train()
        if data_type=='flickr_travel':
            image1, image2, caption1, caption2 = batch[0], batch[1], batch[3], batch[4]
        if data_type=='flickr30k':
            image,caption = batch[0], batch[caption_idx]

        img_embed = imagemodel(image,device,single=False)
        cap_embed = text_model(caption,device,single=False)


        loss=criterion(img_embed,cap_embed,batch[0].shape[0]) + cont_loss(cap_embed,img_embed,batch[0].shape[0])
        #loss=0.1*criterion(img_embed,cap_embed) 
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
        
    return round(epoch_loss,4)


def fine_tune_eval(data_loader, imagemodel,text_model,data_type,device,criterion, caption_idx=None):
    loss_epoch=0
    for idx, batch in enumerate(data_loader):
        imagemodel.eval()
        text_model.eval()
        if data_type=='flickr_travel':
            image1, image2, caption1, caption2 = batch[0], batch[1], batch[3], batch[4]
        if data_type=='flickr30k':
            image,caption = batch[0], batch[caption_idx]

        img_embed = imagemodel(image,device,single=False)
        cap_embed = text_model(caption,device,single=False)


        loss=criterion(img_embed,cap_embed,batch[0].shape[0]) + cont_loss(cap_embed,img_embed,batch[0].shape[0])
        #loss=0.1*criterion(img_embed,cap_embed)
        loss_epoch += loss.item()
    epoch_loss = loss_epoch / len(data_loader)
    return round(epoch_loss,4)
def get_img_txt_embed(images,txt1,txt2,txt3,txt4,tx5,image_model,text_model,caption_idx):

    image_embed = image_model(torch.stack(images),device,single=False)
    if caption_idx==1:
        text_embed1 = text_model(txt1,device,single=False)
        return image_embed ,text_embed1
    if caption_idx==2:
        text_embed2 = text_model(txt2,device,single=False)
        return image_embed ,text_embed2
    if caption_idx==3:
        text_embed3 = text_model(txt3,device,single=False)
        return image_embed ,text_embed3
    if caption_idx==4:
        text_embed4 = text_model(txt4,device,single=False)
        return image_embed ,text_embed4
    if caption_idx==5:
        text_embed5 = text_model(txt5,device,single=False)
        return image_embed ,text_embed5
def recall_score_calculate(image_embed,text_embed,top_k):    
    similarities = cosine_sim(image_embed, text_embed)
    topk_indices = torch.topk(similarities, k=top_k, dim=1)[1]
    recalls=[]
    for i, indices in enumerate(topk_indices):
        if i in indices:
            recalls.append(1)
        else:
            recalls.append(0)
    recall_score = sum(recalls) / len(recalls)
    return recall_score


# In[ ]:


total_epochs=100
for epoch in tqdm(range(total_epochs)):
    train_loss=fine_tune_train(train_loader, model_finetime_img,model_finetime_text,'flickr30k',device,cont_loss,
        optimizer_image, optimizer_text, scheduler_image=scheduler_image,scheduler_text=scheduler_text, caption_idx=1)
    val_loss=fine_tune_eval(test_loader, model_finetime_img,model_finetime_text,'flickr30k',device,cont_loss,caption_idx=1)
    image_embed ,text_embed=get_img_txt_embed(images,txt1,txt2,txt3,txt4,
                                              txt5,model_finetime_img,model_finetime_text,caption_idx=1)
    recall_score=recall_score_calculate(image_embed,text_embed,5)
    
    print('epoch',epoch,'trainloss',train_loss,'val_loss',val_loss,'recall_score',recall_score)
torch.save(model_finetime_img.state_dict(), '/work/08629/pradhakr/maverick2/cv_project/flickr30k-images/ft_image_adam.pth')
torch.save(model_finetime_text.state_dict(), '/work/08629/pradhakr/maverick2/cv_project/flickr30k-images/ft_text_adam.pth')

   
if __name__ == '__main__':
    # Parse command-line arguments

    # Call the main function with the parsed arguments
    main()