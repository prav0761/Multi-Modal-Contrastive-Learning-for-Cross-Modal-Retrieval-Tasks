### Multi-Modal Contrastive Learning for Image Similarity and Cross-modal downstream tasks



Repository Structure
--------------------

    |- code/            # all programmatic code relating to the project
    |  +- templates/    # scripts for generating template files
    |
    |- data/            # all data from the study
    |  |- raw_internal/ # raw data generated in-lab or by collaborators, will not be altered
    |  |- raw_external/ # data from third-party sources, databases etc, will not be altered
    |     +- colormaps/ # color palettes used for all figures
    |
    |- doc/             # documentation for the study and other explanatory material
    |  +- paper/        # contains the generated pdf from knitting the markdown file
    |
    |- results          # all output from workflows and analyses
    |  |- figures/      # graphs, designated for manuscript figures
    |  +- pictures/     # diagrams, images, and other non-graph graphics
    |
    |- .gitignore       # files that will not sync to Github
    |- LICENSE          # license
    |- README.md        # the top level description of content
    |- reproduce.Rproj  # contains project information used to customize the behavior of RStudio  
    +- requirements.txt # the requirements file for reproducing the analysis environment
## Captions Download Link
Captions file - https://drive.google.com/file/d/1iYykGUUOKlFhNT8nT_RHTKXUQPcQjQ23/view?usp=share_link
## Once downloaded use this command to decompress and extract the results_20130124.token file.
```
$ tar-xvzf captions_file.tar
```

## Images Download Link
Images file - https://drive.google.com/file/d/1QVBPbPowBJJkrolDPIWHK1i-1PIJNxpk/view?usp=share_link
## Once downloaded use this command to decompress and extract all the images.
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
