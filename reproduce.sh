#!/bin/bash


python3 -m venv myenv
source myenv/bin/activate
pip3 install torch torchvision pillow requests tqdm pytorch_pretrained_bert nltk
deactivate

mkdir data
tar -zxvf flickr_images.tar.gz -C data
tar -zxvf captions.tar.gz -C data