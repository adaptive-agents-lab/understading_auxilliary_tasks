envs=(Asterix-v1 Freeway-v1 Seaquest-v1 SpaceInvaders-v1 Breakout-v1)

for i in {1..2}
do
    for env in "${envs[@]}"
    do
        seed=$RANDOM
        sbatch slurm_minatar_distorted.sh $seed $env-distor-fixed-byol-detach $env 5 "[[byol_mse,1.0]]" True
        sbatch slurm_minatar_distorted.sh $seed $env-distor-fixed-obs-detach $env 5 "[[obs_recon,1.0]]" True
        sbatch slurm_minatar_distorted.sh $seed $env-distor-fixed-dqn-detach $env 5 "[]" True
        sbatch slurm_minatar_distorted.sh $seed $env-distor-fixed-byol $env 5 "[[byol_mse,1.0],[reward_recon,1.0]]" False
        sbatch slurm_minatar_distorted.sh $seed $env-distor-fixed-obs $env 5 "[[obs_recon,1.0],[reward_recon,1.0]]" False
        sbatch slurm_minatar_distorted.sh $seed $env-distor-fixed-dqn $env 5 "[]" False
    done
done
