### Multi-Modal Contrastive Learning for Image Similarity and Cross-modal downstream tasks
## Captions Download Link
captions file - https://drive.google.com/file/d/1iYykGUUOKlFhNT8nT_RHTKXUQPcQjQ23/view?usp=share_link
## Once downloaded use this command to decompress and extract the results_20130124.token file.
```
$ tar-xvzf captions_file.tar
```

## Images Download Link
images file - https://drive.google.com/file/d/1QVBPbPowBJJkrolDPIWHK1i-1PIJNxpk/view?usp=share_link
## Once downloaded use this command to decompress and extract all the images.
```
$ tar-xvzf flickr30-images.tar.gz
```
## Model Weights Link

model weights files- https://drive.google.com/drive/folders/17ilpn03CDwGvVtZCFqHNHNmzPAuqo78L?usp=share_link

## Reproducing commands


```
$ python3 -m venv myenv
$ source myenv/bin/activater
$ git clone https://github.com/prav0761/Multi-Modal.git
$ pip3 install matplotlib torch torchvision pillow requests tqdm pytorch_pretrained_bert nltk
$ python3 eval_reproduce.py --flickr30k_images_dir_path '/work/08629/pradhakr/maverick2/reproduce/test_reproduce/flickr30k-images' --flickr30k_tokens_dir_path '/work/08629/pradhakr/maverick2/reproduce/test_reproduce/results_20130124.token' --image_weights_file '/work/08629/pradhakr/maverick2/reproduce/test_reproduce/image_model_finetune241_30k.pth' --text_weights_file '/work/08629/pradhakr/maverick2/reproduce/test_reproduce/text_model_finetune241_30k.pth'
```
