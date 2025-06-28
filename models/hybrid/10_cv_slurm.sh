#!/bin/sh

#SBATCH --job-name=10cv # the name of the job
#SBATCH --array=0-9 # the 'k's that is passed to the jobs and it is inclusive, e.g. 0-9 means we run 10 Jobs
#SBATCH -p gpu # GPU queue
#SBATCH --gres=gpu:<gpu_name>:<number of gpus>
#SBATCH --cpus-per-task 32
#SBATCH --mem 120G # Memory per CPU
#SBATCH --mail-type END # This sends you an email once your job finishes
#SBATCH -o .../logs-%A-%a.out # replace ... with the path where you want to save the logs


echo Running Task ID $SLURM_ARRAY_TASK_ID

PYTHONPATH=$(pwd) path_to_your_conda_env/bin/python -u .../h2cm/models/hybrid/train_model.py $SLURM_ARRAY_TASK_ID # replace ... with your path to the script

exit