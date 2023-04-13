#!/bin/bash

#SBATCH -J pretrain_4  # Job name
#SBATCH -o ot4.txt       # Name of stdout output file
#SBATCH -e et4.txt      # Name of stderr error file
#SBATCH -p v100           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 10:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A AustinEnergy       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=praveenradhakrishnan076@utexas.edu


python3 /home1/08629/pradhakr/cv_project/main.py --trial_number 4 --intra_projection_dim 128 --inter_projection_dim 1024 --image_learning_rate 0.03 --text_learning_rate 0.03 --momentum 0.9 --temperature 0.07 --weight_decay 0.0001 --optimizer_type sgd --total_epochs 100 --trade_off_ii 1 --trade_off_cc 1 --trade_off_ic 1e-4 --trade_off_ci 1e-4 --batch_size 128 --margin 0.2 --max_violation True --image_layers_to_train layer3   layer4  --text_layers_to_train h.10 h.11 --caption_index_1 0 --caption_index_2 1 --scheduler_status True --flickr30k_images_dir_path '/work/08629/pradhakr/maverick2/cv_project/flickr30k-images' --flickr30k_tokens_dir_path '/work/08629/pradhakr/maverick2/cv_project/flickr30k_captions/results_20130124.token' --graph_save_dir '/home1/08629/pradhakr/cv_project/graphs/30k_graphs_img_capt' --logresults_save_dir_path '/work/08629/pradhakr/maverick2/cv_project/flickr30k_imagecaption'

