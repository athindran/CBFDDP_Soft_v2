#!/bin/bash
#SBATCH --job-name=cbfddp    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=01:20:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=ALL        # send email when job begins
#SBATCH --mail-user=arkumar@princeton.edu

export PYTHONPATH=/home/arkumar/cbfddp_soft
module purge
module load anaconda3/2023.3

#conda info --envs
conda activate /scratch/gpfs/arkumar/.conda/envs/lincing
#conda info --envs
#python /home/arkumar/cbf_ddp_unknown/dubins_reachability_cbf_ddp_unknown.py
bash /home/arkumar/cbfddp_soft/test_scripts.sh
