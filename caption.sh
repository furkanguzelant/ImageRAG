#!/bin/bash
#SBATCH -A EUHPC_A02_031
#SBATCH -p boost_usr_prod
#SBATCH -N 1                            # 1 node
#SBATCH --time 04:00:00                  # format: HH:MM:SS
#SBATCH --ntasks-per-node=4             # 4 tasks out of 32
#SBATCH --gres=gpu:2                    # 1 gpus per node out of 4
#SBATCH --mem=123000                    # memory per node out of 494000MB (481GB)
#SBATCH --job-name=caption           # job name
#SBATCH --error=./logs/caption.err             # standard error file
#SBATCH --output=./logs/caption.out            # standard output file

export HF_HOME=/leonardo_work/EUHPC_A02_031/hfhub2
module load python/3.10.8--gcc--11.3.0
module load gcc
module load cuda
module load openblas
module load openmpi

source /leonardo_work/EUHPC_A02_031/furkan_env/bin/activate

python blip_caption_parallel.py 