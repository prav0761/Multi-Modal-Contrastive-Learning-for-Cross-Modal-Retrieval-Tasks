#!/bin/bash


#SBATCH -J ft242 # Job name
#SBATCH -o otft242.txt       # Name of stdout output file
#SBATCH -e etft242.txt      # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 10:20:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A A-ib1       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=praveenradhakrishnan076@utexas.edu



python3 /home1/08629/pradhakr/cv_project/finetune_main.py --trial_number 242 --batch_size 128 --output_dim 1024 --data_type 'flickr30k' --image_learning_rate  0.0001 --text_learning_rate 0.0001 --momentum 0.9 --temperature 0.07 --weight_decay 0.0001 --optimizer_type sgd --total_epochs 200 --caption_idx_eval 1 --scheduler_status True --flickr30k_images_dir_path '/work/08629/pradhakr/maverick2/cv_project/flickr30k-images' --flickr30k_tokens_dir_path '/work/08629/pradhakr/maverick2/cv_project/flickr30k_captions/results_20130124.token' --logresults_save_dir_path '/work/08629/pradhakr/maverick2/cv_project/flickr30k_finetune_results' --image_weights_file '/work/08629/pradhakr/maverick2/cv_project/flickr30k_imagecaption/image_model24_30k.pth'  --text_weights_file '/work/08629/pradhakr/maverick2/cv_project/flickr30k_imagecaption/text_model24_30k.pth' 
