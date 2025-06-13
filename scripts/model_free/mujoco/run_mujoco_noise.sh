# envs=("cheetah run" "walker run" "hopper hop" "acrobot swingup" "fish swim" "humanoid stand" "humanoid run" "quadruped run" "finger turn_hard" "hopper stand" "humanoid walk" "pendulum swingup" "quadruped walk" "reacher hard" "swimmer swimmer6")
envs=("humanoid stand")

for env in "${envs[@]}"
do
    set -- $env
    # sbatch slurm_mujoco_random_data.sh $1-$2-random-data-byol-detach $1 $2 10 "[[byol_mse,1.0]]"
    # sbatch slurm_mujoco_random_data.sh $1-$2-random-data-obs-detach $1 $2 10 "[[obs_recon,1.0]]"
    # sbatch slurm_mujoco_random_data.sh $1-$2-random-data-dqn-detach $1 $2 10 "[]"
    echo sbatch slurm_mujoco_random_noise.sh $1-$2-random-noise-byol $1 $2 10 "[[byol_mse,1.0]]"
    echo sbatch slurm_mujoco_random_noise.sh $1-$2-random-noise-obs $1 $2 10 "[[obs_recon,1.0]]"
    echo sbatch slurm_mujoco_random_noise.sh $1-$2-random-noise-dqn $1 $2 10 "[]"
done
