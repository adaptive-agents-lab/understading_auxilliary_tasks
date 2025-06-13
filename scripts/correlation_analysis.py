import csv
import glob
import numpy as np
import jax
import jax.numpy as jnp
import tqdm


def compute_correlation_matrix(samples):
    correlation = (samples.T @ samples) / samples.shape[0]
    correlation = correlation
    return correlation


def strip_all_zero(a):
    idx = np.argwhere(np.all(a[..., :] == 0, axis=0))
    return np.delete(a, idx, axis=1)


runs = {
    "swimmer-swimmer6-dqn": "/checkpoint/voelcker/12097053",
    "reacher-hard-dqn": "/checkpoint/voelcker/12097045",
    "quadruped-walk-dqn": "/checkpoint/voelcker/12097037",
    "pendulum-swingup-dqn": "/checkpoint/voelcker/12097029",
    "humanoid-walk-dqn": "/checkpoint/voelcker/12097021",
    "hopper-stand-dqn": "/checkpoint/voelcker/12097013",
    "quadruped-run-dqn": "/checkpoint/voelcker/12096997",
    "finger-turn_hard-dqn": "/checkpoint/voelcker/12097005",
    "humanoid-run-dqn": "/checkpoint/voelcker/12096989",
    "humanoid-stand-dqn": "/checkpoint/voelcker/12096981",
    "fish-swim-dqn": "/checkpoint/voelcker/12096973",
    "acrobot-swingup-dqn": "/checkpoint/voelcker/12096965",
    "hopper-hop-dqn": "/checkpoint/voelcker/12096957",
    "cheetah-run-dqn": "/checkpoint/voelcker/12096941",
    "walker-run-dqn": "/checkpoint/voelcker/12096949",
    "swimmer-swimmer6-byol": "/checkpoint/voelcker/12097051",
    "reacher-hard-byol": "/checkpoint/voelcker/12097043",
    "quadruped-walk-byol": "/checkpoint/voelcker/12097035",
    "pendulum-swingup-byol": "/checkpoint/voelcker/12097027",
    "humanoid-walk-byol": "/checkpoint/voelcker/12097019",
    "finger-turn_hard-byol": "/checkpoint/voelcker/12097003",
    "quadruped-run-byol": "/checkpoint/voelcker/12096995",
    "hopper-stand-byol": "/checkpoint/voelcker/12097011",
    "humanoid-run-byol": "/checkpoint/voelcker/12096987",
    "humanoid-stand-byol": "/checkpoint/voelcker/12096979",
    "acrobot-swingup-byol": "/checkpoint/voelcker/12096963",
    "fish-swim-byol": "/checkpoint/voelcker/12096971",
    "hopper-hop-byol": "/checkpoint/voelcker/12096955",
    "cheetah-run-byol": "/checkpoint/voelcker/12096939",
    "walker-run-byol": "/checkpoint/voelcker/12096947",
}

printer = csv.DictWriter(
    open("correlation_analysis_mujoco.csv", "w"), fieldnames=["env", "draw", "dist"]
)
printer.writeheader()

sample_lengths = []

for env in sorted(
    [
        "swimmer-swimmer6",
        "reacher-hard",
        "quadruped-walk",
        "pendulum-swingup",
        "humanoid-walk",
        "finger-turn_hard",
        "quadruped-run",
        "hopper-stand",
        "humanoid-run",
        "humanoid-stand",
        "acrobot-swingup",
        "fish-swim",
        "hopper-hop",
        "cheetah-run",
        "walker-run",
    ]
):
    print(env)
    all_samples = []
    for model in ["dqn", "byol"]:
        run = f"{env}-{model}"
        files = glob.glob(f"{runs[run]}/checkpoint/replay*")
        for files in tqdm.tqdm(files[:1]):
            data = np.load(files)["s"]
            all_samples.append(np.load(files)["s"])
    all_samples = np.concatenate(all_samples, axis=1)
    all_samples = all_samples.reshape(-1, all_samples.shape[-1])

    sample_lengths.append((env, all_samples.shape[-1]))

    # for i in range(50):
    #     idx = np.random.choice(all_samples.shape[0], 20000)
    #     # print(np.median(np.linalg.norm(samples[idx], axis=0)))
    #     samples = all_samples[idx]  # - np.mean(all_samples[idx], axis=0, keepdims=True)
    #     samples = samples / np.mean(np.linalg.norm(samples, axis=1))

    #     samples = strip_all_zero(samples)
    #     correlation = jax.jit(compute_correlation_matrix)(jnp.array(samples))
    #     print(correlation.shape)
    #     main_diagonal_elements = np.diag(correlation).reshape(-1)
    #     off_diagonal_elements = correlation[
    #         ~np.eye(correlation.shape[0], dtype=bool)
    #     ].reshape(-1)
    #     min_corr = np.min(correlation)
    #     max_corr = np.max(correlation)

    #     eigv = jnp.linalg.eigh(correlation - jnp.eye(correlation.shape[0])).eigenvalues
    #     print(f"Spectral norm from identity: {jnp.max(eigv)}")
    #     print(
    #         f"Mean distance from identity: {np.mean(np.square(correlation - np.eye(correlation.shape[0])))}"
    #     )
    #     print(
    #         f"Mean distance main diagonal from 1: {np.mean(np.square(main_diagonal_elements - 1.))}"
    #     )
    #     print(
    #         f"Mean distance off diagonal from 0: {np.mean(np.square(off_diagonal_elements))}"
    #     )

    #     corr_mean_hist, corr_mean_bins = np.histogram(main_diagonal_elements, bins=500)
    #     corr_var_hist, corr_var_bins = np.histogram(off_diagonal_elements, bins=500)

    #     printer.writerow(
    #         {
    #             "env": env,
    #             "draw": i,
    #             "dist": np.mean(np.square(correlation - np.eye(correlation.shape[0]))),
    #         }
    #     )

print(list(reversed(sorted(sample_lengths, key=lambda x: x[1]))))
