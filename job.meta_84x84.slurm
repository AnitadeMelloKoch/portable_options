#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=40
#SBATCH --mem=230G
#SBATCH --time=23:00:00
#SBATCH --partition=pancake
#SBATCH --job-name=meta_full_150
#SBATCH --error=meta_full.%J.err
#SBATCH --output=meta_full.%J.out
#SBATCH --gres gpu:2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate port-options

export OMPI_MCA_btl_tcp_if_include=10.42.0.0/24

mpirun -n 6 python -um experiments.divdis_minigrid.divdis_image_meta_full --config_file configs/divdis_image_meta_full_150.gin --base_dir runs --seed 1


