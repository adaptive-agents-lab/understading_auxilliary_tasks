#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH --cpus-per-task=8     # number of cpus required per task
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --qos=scavenger
#SBATCH --time=96:00:00      # time limit
#SBATCH --partition=rtx6000,a40
#SBATCH --job-name=lam-mujoco
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@10
#SBATCH --exclude=gpu062,gpu138
#SBATCH -o /checkpoint/voelcker/%j/slurm.out # STDOUT
#SBATCH --open-mode=append

name=$1
domain_name=$2
task_name=$3
num_envs=$4
loss_functions=$5
detach_enc_critic=$6
checkpoint_dir=${7-"/checkpoint/voelcker/$SLURM_JOB_ID"}

# trap handler - resubmit ourselves
handler()
{
    echo "function handler called at $(date)"
    # do whatever cleanup you want here;
    # checkpoint, sync, etc
    sbatch ${BASH_SOURCE[0]} $name \
        $domain_name \
        $task_name \
        $num_envs \
        $loss_functions \
        $detach_enc_critic \
        $checkpoint_dir
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

cd ~/Code/project_codebases/lambda-ac-mujoco

source ~/venvs/lambda-jax/bin/activate

hostname
nvidia-smi


python aux_tasks/main.py \
    --config-name=main \
    train.seed=$RANDOM \
    name=$name \
    env=dmc_noise \
    env.noise_distraction=True \
    env.name="$domain_name-$task_name" \
    env.domain_name=$domain_name \
    env.task_name=$task_name \
    train.num_seeds=$num_envs \
    algo.loss_functions=$loss_functions \
    hydra.run.dir=$checkpoint_dir \
    hydra.job.chdir=True & wait
