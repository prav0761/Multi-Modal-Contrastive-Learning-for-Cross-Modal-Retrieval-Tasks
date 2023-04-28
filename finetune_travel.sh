#!/bin/bash


#SBATCH -J ft1travel # Job name
#SBATCH -o otft1travel.txt       # Name of stdout output file
#SBATCH -e etft1travel.txt      # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A A-ib1       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=praveenradhakrishnan076@utexas.edu



python3 /home1/08629/pradhakr/cv_project/finetune_main_travel.py --trial_number 1 --batch_size 128 --output_dim 1024 --data_type 'flickr_travel' --image_learning_rate 0.001  --text_learning_rate 0.001 --momentum 0.9 --temperature 0.1 --weight_decay 0.0001 --optimizer_type sgd --total_epochs 100 --caption_idx_eval 1 --scheduler_status True --flickr30k_root_train_dir_path 'data/' --flickr30k_train_dir_path 'data/train' --flickr30k_root_test_dir_path 'data/' --flickr30k_test_dir_path 'data/test' --logresults_save_dir_path '/work/08629/pradhakr/maverick2/cv_project/flickr30k_finetune_results' --image_weights_file '/work/08629/pradhakr/maverick2/cv_project/flickr30k_imagecaption/image_model6_30k.pth'  --text_weights_file '/work/08629/pradhakr/maverick2/cv_project/flickr30k_imagecaption/text_model16_30k.pth' 
