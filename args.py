
# coding: utf-8

# In[5]:


import argparse

def args_c():
    parser = argparse.ArgumentParser(description='Image Captioning Training')
    
    # Model hyperparameters
    parser.add_argument('--trial_number', type=int, default=3, help='Trial number for the experiment')
    parser.add_argument('--intra_projection_dim', type=int, default=128, help='Intra-attention projection dimension')
    parser.add_argument('--inter_projection_dim', type=int, default=1024, help='Inter-attention projection dimension')
    parser.add_argument('--image_learning_rate', type=float, default=0.03, help='Learning rate for image encoder')
    parser.add_argument('--text_learning_rate', type=float, default=1e-4, help='Learning rate for text encoder')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for softmax in contrastive loss')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for SGD optimizer')
    parser.add_argument('--optimizer_type', type=str, default='sgd', help='Optimizer type (sgd or adam)')
    parser.add_argument('--total_epochs', type=int, default=100, help='Total number of training epochs')
    parser.add_argument('--trade_off_ii', type=float, default=1, help='Trade-off for image-image similarity loss')
    parser.add_argument('--trade_off_cc', type=float, default=1, help='Trade-off for caption-caption similarity loss')
    parser.add_argument('--trade_off_ic', type=float, default=1e-4, help='Trade-off for image-caption similarity loss')
    parser.add_argument('--trade_off_ci', type=float, default=1e-4, help='Trade-off for caption-image similarity loss')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--margin', type=float, default=0.2, help='Margin for contrastive loss')
    parser.add_argument('--max_violation', type=bool, default=True, help='Whether to use max violation or not')
    parser.add_argument('--image_layers_to_train', nargs='+', default=['layer3', 'layer4'], help='Image encoder layers to train')
    parser.add_argument('--text_layers_to_train', nargs='+', default=['h.10', 'h.11'], help='Text encoder layers to train')
    parser.add_argument('--caption_index_1', type=int, default=0, help='Index of first caption to be compared')
    parser.add_argument('--caption_index_2', type=int, default=1, help='Index of second caption to be compared')
    
    # Paths and directories
    parser.add_argument('--scheduler_status', type=bool, default=True, help='Whether to use scheduler or not')
    parser.add_argument('--flickr30k_images_dir_path', 
                        type=str, default='/work/08629/pradhakr/maverick2/cv_project/flickr30k-images', 
                        help='Directory path for Flickr30k images')
    parser.add_argument('--logresults_save_dir_path', 
                        type=str, default='/work/08629/pradhakr/maverick2/cv_project/flickr30k_imagecaption', 
                        help='Directory path for Flickr30k images')
    parser.add_argument('--flickr30k_tokens_dir_path', type=str, 
                        default='/work/08629/pradhakr/maverick2/cv_project/flickr30k_captions/results_20130124.token',
                        help='Directory path for Flickr30k captions')
    parser.add_argument('--graph_save_dir', type=str, default='/work/08629/pradhakr/maverick2/cv_project/flickr30k_imagecaption',
                        help='where to save your graphs')
    
    #flickr30k_images_dir_path='/work/08629/pradhakr/maverick2/cv_project/flickr30k-images'
    #flickr30k_tokens_dir_path='/work/08629/pradhakr/maverick2/cv_project/flickr30k_captions/results_20130124.token'
    #graph_save_dir = '/home1/08629/pradhakr/cv_project/graphs/30k_graphs_img_capt'
    #logresults_save_dir_path='/work/08629/pradhakr/maverick2/cv_project/flickr30k_imagecaption'                     
                        
    args = parser.parse_args()

    return args


def args_finetune():
    parser = argparse.ArgumentParser(description='finetuning Training')
    
    # Model hyperparameters
    parser.add_argument('--trial_number', type=int, default=6, help='Trial number for the experiment')
    parser.add_argument('--output_dim', type=int, default=1024, help='Inter-attention projection dimension')
    parser.add_argument('--data_type', type=str, default='flickr30k', help='flickrtravel or flickr30k')
    parser.add_argument('--image_learning_rate', type=float, default=0.001, help='Learning rate for image encoder')
    parser.add_argument('--text_learning_rate', type=float, default=0.001, help='Learning rate for text encoder')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for softmax in contrastive loss')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for SGD optimizer')
    parser.add_argument('--optimizer_type', type=str, default='sgd', help='Optimizer type (sgd or adam)')
    parser.add_argument('--total_epochs', type=int, default=100, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--caption_idx_eval', type=int, default=1, help='Index of first caption to be compared')
    
    # Paths and directories
    parser.add_argument('--scheduler_status', type=bool, default=True, help='Whether to use scheduler or not')
    parser.add_argument('--flickr30k_images_dir_path', 
                        type=str, default='/work/08629/pradhakr/maverick2/cv_project/flickr30k-images', 
                        help='Directory path for Flickr30k images')
    parser.add_argument('--logresults_save_dir_path', 
                        type=str, default='/work/08629/pradhakr/maverick2/cv_project/flickr30k_finetune_results', 
                        help='Directory path for Flickr30k results finetuning')
    parser.add_argument('--flickr30k_tokens_dir_path', type=str, 
                        default='/work/08629/pradhakr/maverick2/cv_project/flickr30k_captions/results_20130124.token',
                        help='Directory path for Flickr30k captions')
    parser.add_argument('--image_weights_file', type=str, 
                        default='/work/08629/pradhakr/maverick2/cv_project/flickr30k_imagecaption/image_model6_30k.pth',
                        help='image_weights_file')                   
    parser.add_argument('--text_weights_file', type=str, 
                        default='/work/08629/pradhakr/maverick2/cv_project/flickr30k_imagecaption/text_model6_30k.pth',
                        help='text_weights_file')                    
    args = parser.parse_args()

    return args
