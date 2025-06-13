envs=("cheetah run" "walker run" "hopper hop" "acrobot swingup" "fish swim" "humanoid stand" "humanoid run" "quadruped run" "finger turn_hard" "hopper stand" "humanoid walk" "pendulum swingup" "quadruped walk" "reacher hard" "swimmer swimmer6")

for env in "${envs[@]}"
do
    set -- $env
    sbatch slurm_mujoco.sh $1-$2-byol-detach $1 $2 10 "[[byol_mse,1.0]]" True
    sbatch slurm_mujoco.sh $1-$2-obs-detach $1 $2 10 "[[obs_recon,1.0]]" True
    sbatch slurm_mujoco.sh $1-$2-dqn-detach $1 $2 10 "[]" True
    sbatch slurm_mujoco.sh $1-$2-byol-rew-detach $1 $2 10 "[[byol_mse,1.0],[reward_recon,1.0]]" True
    sbatch slurm_mujoco.sh $1-$2-obs-rew-detach $1 $2 10 "[[obs_recon,1.0],[reward_recon,1.0]]" True
    sbatch slurm_mujoco.sh $1-$2-byol $1 $2 10 "[[byol_mse,1.0],[reward_recon,1.0]]" False
    sbatch slurm_mujoco.sh $1-$2-obs $1 $2 10 "[[obs_recon,1.0],[reward_recon,1.0]]" False
    sbatch slurm_mujoco.sh $1-$2-dqn $1 $2 10 "[]" False
done
