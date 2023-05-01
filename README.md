### Multi-Modal Contrastive Learning for Image Similarity and Cross-modal downstream tasks



Repository Structure
--------------------

    |- args/            # args
    |- captions_transform/# scripts for text augmentations
    |- image_transform/# scripts for image augmentations
    |- dataset.py/ # scripts for dataset loaders
    |- eval_reproduce.py # scripts for reproducing results and evaluation
    |- finetune_main.py  # scripts for finetuning on flickr30k_dataset for cross modal tasks
    |- finetune_main_travel.py  # scripts for finetuning on flickr travel dataset for visaluzations img-sim, txt-img retrieval
    |- lars.py          # lars optimizer
    |- logger.py      # scripts for logger functions
    |- main.py   # scripts for pretraining on flickr30k(main process)
    |- metrics.py # scripts for loss functions, optimizer functions
    |- models.py       # backbone models and finetune models
    |- LICENSE          # license
    |- README.md        # the top level description of content and commands to reproduce results, data download instructions
    |- train_fns.py  # contains scripts for training, validation functions
    |- utils.py #   # scripts for helper functions and metrics calculation code
    
    
## Data Downloading Instructions
Please use the below given google drive links for downloading captions file, images, weights files seperately. Once you downloaded all the files to your local desktop, if you are using tacc i would suggest to move all the weights files, data files to your work or scratch dir since home directory wont have enough space. Once you have these files in tacc directories, just specify the directories in final eval.py command(mentioned in last) to reproduce the result

## GPU
Please run of v100 or p100 in maverick2 since gtx will have memory issues, in frontera you can run in all devices.
## Captions Download Link
Captions file - https://drive.google.com/file/d/1iYykGUUOKlFhNT8nT_RHTKXUQPcQjQ23/view?usp=share_link
Once downloaded use this command to decompress and extract the results_20130124.token file.
```
$ tar-xvzf captions_file.tar
```

## Images Download Link
Images file - https://drive.google.com/file/d/1QVBPbPowBJJkrolDPIWHK1i-1PIJNxpk/view?usp=share_link
Once downloaded use this command to decompress and extract all the images.
```
$ tar-xvzf flickr30-images.tar.gz
```
## Model Weights Link
This link has weights files for both image and text encoder

Model weights files- https://drive.google.com/drive/folders/17ilpn03CDwGvVtZCFqHNHNmzPAuqo78L?usp=share_link

## Reproducing results for image-text and text-image R@1,R@5,R@10 recall score on validation set
Replace the directories in the code with your current locations of data and weights files.
```
$ python3 -m venv myenv
$ source myenv/bin/activate
$ git clone https://github.com/prav0761/Multi-Modal.git
$ pip3 install matplotlib torch torchvision pillow requests tqdm pytorch_pretrained_bert nltk
$ python3 eval_reproduce.py --flickr30k_images_dir_path '/work/08629/pradhakr/maverick2/reproduce/test_reproduce/flickr30k-images' --flickr30k_tokens_dir_path '/work/08629/pradhakr/maverick2/reproduce/test_reproduce/results_20130124.token' --image_weights_file '/work/08629/pradhakr/maverick2/reproduce/test_reproduce/image_model_finetune241_30k.pth' --text_weights_file '/work/08629/pradhakr/maverick2/reproduce/test_reproduce/text_model_finetune241_30k.pth'
```
