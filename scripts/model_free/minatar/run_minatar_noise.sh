envs=(Asterix-v1 Freeway-v1 Seaquest-v1 SpaceInvaders-v1 Breakout-v1)

# for i in {1..5}
# do
#     for env in "${envs[@]}"
#     do
#         sbatch slurm_minatar_random_data.sh $env-byol-noise-data $env 2 "[[byol_mse,1.0],[reward_recon,1.0]]"
#         sbatch slurm_minatar_random_data.sh $env-obs-noise-data $env 2 "[[obs_recon,1.0],[reward_recon,1.0]]"
#         sbatch slurm_minatar_random_data.sh $env-dqn-noise-data $env 2 "[]"
#     done
# done
for i in {1..5}
do
    for env in "${envs[@]}"
    do
        sbatch slurm_minatar_random_noise.sh $env-byol-noise-random $env 2 "[[byol_mse,1.0],[reward_recon,1.0]]"
        sbatch slurm_minatar_random_noise.sh $env-obs-noise-random $env 2 "[[obs_recon,1.0],[reward_recon,1.0]]"
        sbatch slurm_minatar_random_noise.sh $env-dqn-noise-random $env 2 "[]"
    done
done
