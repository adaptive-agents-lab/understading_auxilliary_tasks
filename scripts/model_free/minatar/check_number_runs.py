import os

source_dir = "/checkpoint/voelcker/12109414/"

for env in ["Asterix-v1", "Freeway-v1", "Breakout-v1", "Seaquest-v1", "SpaceInvaders-v1"]:
    for job_name in ["-obs", "-dqn", "-byol", "-obs-detach", "-obs-rew-detach", "-byol-detach", "-byol-rew-detach", "-dqn-detach", "-obs-noise-random", "-dqn-noise-random", "-byol-noise-random", "-obs-noise-data", "-dqn-noise-data", "-byol-noise-data", "-distor-fixed-obs", "-distor-fixed-dqn", "-distor-fixed-byol"]:
        run_path = source_dir + env + "/" + job_name

        _, _, files = next(os.walk(run_path))
        file_count = len(files)
        if file_count < 10:
            print(run_path)
            print(file_count)
