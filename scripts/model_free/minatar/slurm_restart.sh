#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH --cpus-per-task=8     # number of cpus required per task
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --qos=scavenger
#SBATCH --time=96:00:00      # time limit
#SBATCH --partition=rtx6000,a40
#SBATCH --job-name=lam-main
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@10
#SBATCH --exclude=gpu062,gpu138
#SBATCH -o /checkpoint/voelcker/%j/slurm.out # STDOUT
#SBATCH --open-mode=append


# trap handler - resubmit ourselves
handler()
{
    echo "function handler called at $(date)"
    sbatch slurm_restart.sh $1
}
# register signal handler
trap handler SIGUSR1

source ~/.bashrc

module load cuda-11.8

export MUJOCO_PY_BYPASS_LOCK=True
export LD_LIBRARY_PATH=/pkgs/cudnn-8.8/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/pkgs/cuda-11.8/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
export MUJOCO_GL=egl 
export HYDRA_FULL_ERROR=1

cd ~/Code/project_codebases/lambda-ac-jax

source ~/venvs/lambda-jax/bin/activate

eval "$1" & wait
