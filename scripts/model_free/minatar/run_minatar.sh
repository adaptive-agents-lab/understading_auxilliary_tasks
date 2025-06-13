envs=(Asterix-v1 Freeway-v1 Seaquest-v1 SpaceInvaders-v1 Breakout-v1)

for i in {1..2}
do
    for env in "${envs[@]}"
    do
        # sbatch slurm_minatar.sh $env-byol-detach $env 5 "[[byol_mse,1.0]]" True
        # sbatch slurm_minatar.sh $env-obs-detach $env 5 "[[obs_recon,1.0]]" True
        # sbatch slurm_minatar.sh $env-dqn-detach $env 5 "[]" True
        # sbatch slurm_minatar.sh $env-byol-rew-detach $env 5 "[[byol_mse,1.0],[reward_recon,1.0]]" True
        # sbatch slurm_minatar.sh $env-obs-rew-detach $env 5 "[[obs_recon,1.0],[reward_recon,1.0]]" True
        sbatch slurm_minatar.sh $env-data-byol $env 5 "[[byol_mse,1.0],[reward_recon,1.0]]" False
        sbatch slurm_minatar.sh $env-data-obs $env 5 "[[obs_recon,1.0],[reward_recon,1.0]]" False
        sbatch slurm_minatar.sh $env-data-dqn $env 5 "[]" False
    done
done
