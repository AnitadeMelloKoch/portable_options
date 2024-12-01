#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --error=<clip_dog_eval>.%J.err
#SBATCH --output=<clip_dog_eval>.%J.out
#SBATCH --mail-type=END
#SBATCH --mail-user=yuechuan_yang@brown.edu

module load miniconda3/23.11.0s

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh 
conda activate divlis
python -m evaluation.dog_classifier_evaluation --base_dir runs/ --seed 100 --config_file configs/dog_evaluator_classifier.gin