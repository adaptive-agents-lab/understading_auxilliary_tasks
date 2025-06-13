envs=("cheetah run" "walker run" "hopper hop" "acrobot swingup" "fish swim" "humanoid stand" "humanoid run" "quadruped run" "finger turn_hard" "hopper stand" "humanoid walk" "pendulum swingup" "quadruped walk" "reacher hard" "swimmer swimmer6")

for env in "${envs[@]}"
do
    set -- $env
    seed=$RANDOM
    sbatch slurm_mujoco_distorted.sh $seed $1-$2-byol-distorted-fixed $1 $2 10 "[[byol_mse,1.0]]"
    sbatch slurm_mujoco_distorted.sh $seed $1-$2-obs-distorted-fixed $1 $2 10 "[[obs_recon,1.0]]"
    sbatch slurm_mujoco_distorted.sh $seed $1-$2-dqn-distorted-fixed $1 $2 10 "[]"
done
