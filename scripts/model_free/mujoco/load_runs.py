import wandb
import os

api = wandb.Api()

# Specify your filters based on the config attribute

for env in [
    "cheetah-run",
    "walker-run",
    "hopper-hop",
    "acrobot-swingup",
    "fish-swim",
    "humanoid-stand",
    "humanoid-run",
    "quadruped-run",
    "finger-turn_hard",
    "hopper-stand",
    "humanoid-walk",
    "pendulum-swingup",
    "quadruped-walk",
    "reacher-hard",
    "swimmer-swimmer6",
]:
    for job_name in [
        "-byol",
        "-byol-random-data",
        "-distor-fixed-byol",
        "-distorted-obs",
        "-dqn-noise-random",
        "-obs",
        "-obs-random-data",
        "-random-data-byol-detach",
        "-random-noise-obs",
        "-byol-detach",
        "-byol-random-data-detach",
        "-distor-fixed-dqn",
        "-dqn",
        "-dqn-random-data",
        "-obs-detach",
        "-obs-random-data-detach",
        "-random-data-dqn-detach",
        "-byol-distorted-fixed",
        "-byol-random-noise",
        "-distor-fixed-obs",
        "-dqn-detach",
        "-dqn-random-data-detach",
        "-obs-distorted-fixed",
        "-obs-random-noise",
        "-random-data-obs-detach",
        "-byol-noise-data",
        "-byol-random-noise-detach",
        "-distorted-byol",
        "-dqn-distorted-fixed",
        "-dqn-random-noise",
        "-obs-noise-data",
        "-obs-random-noise-detach",
        "-random-noise-byol",
        "-byol-noise-random",
        "-byol-rew-detach",
        "-distorted-dqn",
        "-dqn-noise-data",
        "-dqn-random-noise-detach",
        "-obs-noise-random",
        "-obs-rew-detach",
        "-random-noise-dqn",
    ]:
        run_name = env + job_name

        config_filters = {
            "config.name": run_name,
            "config.env.name": env,
        }

        # Get runs that match the specified filters
        runs = api.runs("viper_svg/aux_tasks", filters=config_filters)

        for run in runs:
            if run.state == "finished":
                run_name = env + job_name

                config_filters = {
                    "config.name": run_name,
                    "config.env.name": env,
                    # Add more filters as needed
                }

                # Get runs that match the specified filters
                runs = api.runs("viper_svg/aux_tasks", filters=config_filters)

                keys = [[f"ep_rew/{i}"] for i in range(5)]  # for i in range(10)]

                run_returns = {}

                dir_name = f"/checkpoint/voelcker/{os.environ['SLURM_JOB_ID']}/{env}/{job_name}/"
                os.makedirs(dir_name, exist_ok=True)

                for run in runs:
                    any_short = False
                    # Download logs or perform other actions
                    if run.state == "finished":
                        for k in keys:
                            # Iterate through the runs and download logs
                            history = run.history(keys=k, samples=100000)
                            length = history.to_numpy().shape[0]
                            if length == 0:
                                continue
                            history.to_csv(
                                f"{dir_name}{run.config['name']}-{k[0].replace('/', '')}-{run.name}.csv"
                            )
                            any_short = any_short or length < 409
                    if any_short:
                        print(
                            f"Caution! /checkpoint/voelcker/{os.environ['SLURM_JOB_ID']}/{run.config['name']}-{run.name}.csv had length {length}"
                        )
                    else:
                        print(
                            f"Processed /checkpoint/voelcker/{os.environ['SLURM_JOB_ID']}/{run.config['name']}-{run.name}.csv"
                        )
