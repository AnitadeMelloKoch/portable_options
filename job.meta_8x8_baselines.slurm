#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=230G
#SBATCH --time=23:00:00
#SBATCH --partition=pancake
#SBATCH --job-name=base_meta_full_10_dqn
#SBATCH --error=base_meta_full_10.%J.err
#SBATCH --output=base_meta_full_10.%J.out
#SBATCH --gres gpu:2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate port-options

export OMPI_MCA_btl_tcp_if_include=10.42.0.0/24

mpirun -n 1 python -um experiments.divdis_minigrid.baseline_dqn --config_file configs/baseline_minigrid_dqn.gin --base_dir runs --seed 22




