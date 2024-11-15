#!/bin/sh
#SBATCH --job-name=rbc
#SBATCH --output=/iitdh/PhD/amitkumar/slurm-%j.out
#SBATCH --error=/iitdh/PhD/amitkumar/slurm-%j.err
#SBATCH --ntasks=64
#SBATCH --time=12:00:00
#SBATCH --partition=long
#SBATCH --mail-user=173031003@iitdh.ac.in
#SBATCH --mail-type=ALL

mpirun -n 64 /apps/FDS/bin/./fds

