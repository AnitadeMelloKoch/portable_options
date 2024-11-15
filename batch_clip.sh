#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --error=<clip>.%J.err
#SBATCH --output=<clip>.%J.out

module load miniconda3/23.11.0s

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh 
conda activate divlis
python -m experiments.dog.dog_classification --base_dir runs/ --seed 100 --config_file configs/dog_classification.gin
