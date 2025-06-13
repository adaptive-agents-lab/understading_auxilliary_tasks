import wandb
import os

api = wandb.Api()

# Specify your filters based on the config attribute

env = "SpaceInvaders-v1"
job_name = "-dqn-noise-random"
distr=2
run_name = env + job_name

config_filters = {
    "config.name": run_name,
    "config.env.name": env,
    "config.env.num_distractions": distr
        # Add more filters as needed
}

# Get runs that match the specified filters
runs = api.runs("viper_svg/aux_tasks", filters=config_filters)

keys = [[f"ep_rew/{i}"] for i in range(5)] # for i in range(10)]

run_returns = {}

dir_name = f"/checkpoint/voelcker/{os.environ['SLURM_JOB_ID']}/{env}/{job_name}/"
os.makedirs(dir_name, exist_ok=True)

for run in runs:
    any_short = False
    # Download logs or perform other actions
    for k in keys:
        # Iterate through the runs and download logs
        history = run.history(keys=k,samples=100000)
        length = history.to_numpy().shape[0]
        if length == 0:
            continue
        history.to_csv(f"{dir_name}{run.config['name']}-{k[0].replace('/', '')}-{run.name}.csv")
        any_short = any_short or length < 4950
    if any_short:
        print(f"Caution! /checkpoint/voelcker/{os.environ['SLURM_JOB_ID']}/{run.config['name']}-{run.name}.csv had length {length}")
    else:
        print(f"Processed /checkpoint/voelcker/{os.environ['SLURM_JOB_ID']}/{run.config['name']}-{run.name}.csv")
