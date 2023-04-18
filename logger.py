
# coding: utf-8

# In[ ]:
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

class Logger:
    def __init__(self, log_file, 
                 resnet_save_file, 
                 gpt_save_file, 
                 optimizer, 
                 image_learning_rate, 
                 text_learning_rate,
                 weight_decay,
                 batch_size,
                 Momentum, 
                 temperature, 
                 total_epochs,
                 trade_off_cc,
                 trade_off_ii,
                 trade_off_ic,
                 trade_off_ci,
                 resnet_layers,
                 gpt_layers,
                 intra_projection_dim,
                 inter_projection_dim,
                 scheduler,
                 margin,
                 max_violation):
        """
        Initializes a Logger object with the given parameters.,

        Args:
            log_file (str): path to the log file
            resnet_save_file (str): path to save the ResNet model
            gpt_save_file (str): path to save the GPT model
            optimizer (str): name of the optimizer used for training
            learning_rate (float): learning rate used for training
            weight_decay (float): weight decay used for training
            batch_size (int): batch size used for training
            Momentum (float): momentum used for training
            temperature (float): temperature used for training
            total_epochs (int): total number of epochs to run
        """
        self.log_file = log_file
        self.train_losses = []  # list to store train losses for each epoch
        self.val_losses = []  # list to store validation losses for each epoch
        self.resnet_save_file = resnet_save_file
        self.gpt_save_file = gpt_save_file
        self.best_val_loss = float('inf')  # initialize the best validation loss as infinity
        self.best_epoch = -1  # initialize the best epoch to -1
        self.message = ""
        self.message += f'total epochs: {total_epochs:.1f}\n'
        self.message += f'Optimizer: {optimizer}\n'
        self.message += f'image_learning_rate: {image_learning_rate:.4f}\n'
        self.message += f'text_learning_rate: {text_learning_rate:.6f}\n'
        self.message += f' Scheduler: {scheduler}\n'
        self.message += f'Weight decay: {weight_decay:.4f}\n'
        self.message += f'Batch size: {batch_size}\n'
        self.message += f'Momentum: {Momentum}\n'
        self.message += f'temperature: {temperature}\n'
        self.message += f' trade_off_cc: {trade_off_cc}\n'
        self.message += f' trade_off_ii: {trade_off_ii}\n'
        self.message += f' trade_off_ic: {trade_off_ic}\n'
        self.message += f' trade_off_ci: {trade_off_ci}\n'
        self.message += f' resnet_layers: {resnet_layers}\n'
        self.message += f' gpt_layers: {gpt_layers}\n'
        self.message += f' margin: {margin}\n'
        self.message += f' max_violation: {max_violation}\n'
        self.message += f' intra_projection_dim: {intra_projection_dim}\n'
        self.message += f' inter_projection_dim: {inter_projection_dim}\n\n'
        
        with open(self.log_file, 'w') as f:
            f.write(self.message)
    def log(self, epoch, resnet_model, gpt_model, train_loss, test_loss, time_for_epoch):
        """
        Logs the training progress and saves the models with the best validation loss.

        Args:
        epoch: int, the current epoch number.
        resnet_model: nn.Module, the ResNet model.
        gpt_model: nn.Module, the GPT model.
        train_loss: float, the training loss.
        test_loss: float, the validation loss.
        time_for_epoch: float, the time taken for this epoch.
        """
        message = f'Epoch {epoch}: Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Time:{round(time_for_epoch,2)}\n'

        with open(self.log_file, 'a') as f:
            f.write(message)
            
        self.train_losses.append(train_loss)
        self.val_losses.append(test_loss)

        if test_loss < self.best_val_loss:
            torch.save(resnet_model.state_dict(), self.resnet_save_file)
            
            torch.save(gpt_model.state_dict(), self.gpt_save_file)

            self.best_val_loss = test_loss
            self.best_epoch = epoch  # update best_epoch

    def start_training(self):
        """
        Logs the start of training.
        """
        self.start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.message = f'Starting training at {self.start_time}\n\n'  # Create a message indicating the start time of training
        with open(self.log_file, 'a') as f:
            f.write(self.message)  # Write the message to the log file


    def end_training(self):
        """
        Logs the end of training and the best validation loss achieved with the corresponding epoch.
        """
        self.end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.message = f'Ending training at {self.end_time}\n'
        self.message += f'Best validation loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})\n\n'
        with open(self.log_file, 'a') as f:
            f.write(self.message)

    def plot_losses(self,
                    trial_number, 
                    save_dir,
                    optimizer,
                    image_learning_rate,
                    text_learning_rate,
                    weight_decay, 
                    batch_size,
                    Momentum,
                    temperature,
                    total_epochs,
                    trade_off_cc,
                    trade_off_ii,
                    trade_off_ci, 
                    trade_off_ic,
                    resnet_layers,
                    gpt_layers,
                    intra_projection_dim,
                    inter_projection_dim,
                    scheduler,
                     margin,
                   max_violation):
        """
        Plots the train and validation losses and saves the plot to a file.

        Args:
        - trial_number (int): the trial number to include in the plot title
        - save_dir (str): the directory to save the plot to
        """
        fig, ax = plt.subplots(figsize=(15,5))
        ax.plot(range(len(self.train_losses)), self.train_losses, label='Train Loss')
        ax.plot(range(len(self.val_losses)), self.val_losses, label='Validation Loss')
        ax.annotate(f'text_learning_rate: {image_learning_rate:.6f}', xy=(0.6, 1), xycoords='axes fraction')
        ax.annotate(f'total epochs: {total_epochs:.1f}', xy=(0.6, 0.95), xycoords='axes fraction')
        ax.annotate(f'Optimizer: {optimizer}', xy=(0.6, 0.9), xycoords='axes fraction')
        ax.annotate(f'image_learning_rate: {image_learning_rate:.4f}', xy=(0.6, 0.85), xycoords='axes fraction')
        ax.annotate(f'Scheduler: {scheduler}', xy=(0.6, 0.8), xycoords='axes fraction')
        ax.annotate(f'Weight decay: {weight_decay:.4f}', xy=(0.6, 0.75), xycoords='axes fraction')
        ax.annotate(f'Batch size: {batch_size}', xy=(0.6, 0.7), xycoords='axes fraction')
        ax.annotate(f'Momentum: {Momentum}', xy=(0.6, 0.65), xycoords='axes fraction')
        ax.annotate(f'temperature: {temperature}', xy=(0.6, 0.6), xycoords='axes fraction')
        ax.annotate(f'trade_off_cc: {trade_off_cc}', xy=(0.6, 0.55), xycoords='axes fraction')
        ax.annotate(f'trade_off_ii: {trade_off_ii}', xy=(0.6, 0.5), xycoords='axes fraction')
        ax.annotate(f'trade_off_ci: {trade_off_ci}', xy=(0.6, 0.45), xycoords='axes fraction')
        ax.annotate(f'trade_off_ic: {trade_off_ic}', xy=(0.6, 0.4), xycoords='axes fraction')
        ax.annotate(f'resnet_trainable_layers: {resnet_layers}', xy=(0.6, 0.35), xycoords='axes fraction')
        ax.annotate(f'gpt_trainable_layers: {gpt_layers}', xy=(0.6, 0.3), xycoords='axes fraction')
        ax.annotate(f'intra_projection_dim: {intra_projection_dim}', xy=(0.6, 0.25), xycoords='axes fraction')
        ax.annotate(f'inter_projection_dim: {inter_projection_dim}', xy=(0.6, 0.2), xycoords='axes fraction')
        ax.annotate(f'margin: {margin}', xy=(0.6, 0.15), xycoords='axes fraction')
        ax.annotate(f'max_violation: {max_violation}', xy=(0.6, 0.05), xycoords='axes fraction')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'training vs testing Trial-{trial_number}')
        ax.legend(loc='center left')
        file_path = os.path.join(save_dir, f"trial_{trial_number}.png")
        plt.savefig(file_path)
        plt.close()

class Fine_Tune_Logger:
    def __init__(self, 
                 log_file, 
                 resnet_save_file, 
                 gpt_save_file, 
                 optimizer, 
                 image_learning_rate, 
                 text_learning_rate,
                 weight_decay,
                 batch_size,
                 Momentum, 
                 temperature, 
                 total_epochs,
                output_dim,
                 scheduler):
        """
        Initializes a Logger object with the given parameters.,

        Args:
            log_file (str): path to the log file
            resnet_save_file (str): path to save the ResNet model
            gpt_save_file (str): path to save the GPT model
            optimizer (str): name of the optimizer used for training
            learning_rate (float): learning rate used for training
            weight_decay (float): weight decay used for training
            batch_size (int): batch size used for training
            Momentum (float): momentum used for training
            temperature (float): temperature used for training
            total_epochs (int): total number of epochs to run
        """
        self.log_file = log_file
        self.r_5_itscore = []  # list to store validation losses for each epoch
        self.resnet_save_file = resnet_save_file
        self.gpt_save_file = gpt_save_file
        self.best_r_5_itscore = float('-inf')  # initialize the best validation loss as infinity
        self.best_epoch = -1  # initialize the best epoch to -1
        self.message = ""
        self.message += f'total epochs: {total_epochs:.1f}\n'
        self.message += f'Optimizer: {optimizer}\n'
        self.message += f'image_learning_rate: {image_learning_rate:.4f}\n'
        self.message += f'text_learning_rate: {text_learning_rate:.6f}\n'
        self.message += f' Scheduler: {scheduler}\n'
        self.message += f'Weight decay: {weight_decay:.4f}\n'
        self.message += f'Batch size: {batch_size}\n'
        self.message += f'Momentum: {Momentum}\n'
        self.message += f'temperature: {temperature}\n'
        self.message += f' output_dim: {output_dim}\n\n'
        
        with open(self.log_file, 'w') as f:
            f.write(self.message)
    def fine_tune_log(self, epoch, resnet_model, gpt_model, r_5_it, time_for_epoch):
        """
        Logs the training progress and saves the models with the best validation loss.

        Args:
        epoch: int, the current epoch number.
        resnet_model: nn.Module, the ResNet model.
        gpt_model: nn.Module, the GPT model.
        train_loss: float, the training loss.
        test_loss: float, the validation loss.
        time_for_epoch: float, the time taken for this epoch.
        """
        message = f'Epoch {epoch}: R@5_IMAGE-TEXT-SCORE: {r_5_it:.4f}, Time:{round(time_for_epoch,2)}\n'

        with open(self.log_file, 'a') as f:
            f.write(message)
            
        self.r_5_itscore.append(r_5_it)

        if r_5_it > self.best_r_5_itscore:
            torch.save(resnet_model.state_dict(), self.resnet_save_file)
            
            torch.save(gpt_model.state_dict(), self.gpt_save_file)

            self.resnet_model = r_5_it
            self.best_epoch = epoch  # update best_epoch

    def start_training(self):
        """
        Logs the start of training.
        """
        self.start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.message = f'Starting training at {self.start_time}\n\n'  # Create a message indicating the start time of training
        with open(self.log_file, 'a') as f:
            f.write(self.message)  # Write the message to the log file


    def end_training(self):
        """
        Logs the end of training and the best validation loss achieved with the corresponding epoch.
        """
        self.end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.message = f'Ending training at {self.end_time}\n'
        self.message += f'Best validation loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})\n\n'
        with open(self.log_file, 'a') as f:
            f.write(self.message)