import glob
import csv
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
    "SpaceInvaders-v1-dqn": "/checkpoint/voelcker/12043554",
    "SpaceInvaders-v1-byol": "/checkpoint/voelcker/12043514",
    "Seaquest-v1-dqn": "/checkpoint/voelcker/12043506",
    "Seaquest-v1-byol": "/checkpoint/voelcker/12043546",
    "Breakout-v1-dqn": "/checkpoint/voelcker/12043562",
    "Breakout-v1-byol": "/checkpoint/voelcker/12043522",
    "Asterix-v1-dqn": "/checkpoint/voelcker/12043530",
    "Asterix-v1-byol": "/checkpoint/voelcker/12043490",
    "Freeway-v1-dqn": "/checkpoint/voelcker/12043498",
    "Freeway-v1-byol": "/checkpoint/voelcker/12043538",
}
runs = {
    "Asterix-v1-1": "/checkpoint/voelcker/12114395",
    "Asterix-v1-2": "/checkpoint/voelcker/12114396",
    "Asterix-v1-3": "/checkpoint/voelcker/12114397",
    "Freeway-v1-1": "/checkpoint/voelcker/12114398",
    "Freeway-v1-2": "/checkpoint/voelcker/12114399",
    "Freeway-v1-3": "/checkpoint/voelcker/12114400",
    "Seaquest-v1-1": "/checkpoint/voelcker/12114401",
    "Seaquest-v1-2": "/checkpoint/voelcker/12114402",
    "Seaquest-v1-3": "/checkpoint/voelcker/12114403",
    "SpaceInvaders-v1-1": "/checkpoint/voelcker/12114404",
    "SpaceInvaders-v1-2": "/checkpoint/voelcker/12114405",
    "SpaceInvaders-v1-3": "/checkpoint/voelcker/12114406",
    "Breakout-v1-1": "/checkpoint/voelcker/12114407",
    "Breakout-v1-2": "/checkpoint/voelcker/12114408",
    "Breakout-v1-3": "/checkpoint/voelcker/12114409",
    "Asterix-v1-4": "/checkpoint/voelcker/12114410",
    "Asterix-v1-5": "/checkpoint/voelcker/12114411",
    "Asterix-v1-6": "/checkpoint/voelcker/12114412",
    "Freeway-v1-4": "/checkpoint/voelcker/12114413",
    "Freeway-v1-5": "/checkpoint/voelcker/12114414",
    "Freeway-v1-6": "/checkpoint/voelcker/12114415",
    "Seaquest-v1-4": "/checkpoint/voelcker/12114416",
    "Seaquest-v1-5": "/checkpoint/voelcker/12114417",
    "Seaquest-v1-6": "/checkpoint/voelcker/12114418",
    "SpaceInvaders-v1-4": "/checkpoint/voelcker/12114419",
    "SpaceInvaders-v1-5": "/checkpoint/voelcker/12114420",
    "SpaceInvaders-v1-6": "/checkpoint/voelcker/12114421",
    "Breakout-v1-4": "/checkpoint/voelcker/12114422",
    "Breakout-v1-5": "/checkpoint/voelcker/12114423",
    "Breakout-v1-6": "/checkpoint/voelcker/12114424",
}

printer = csv.DictWriter(
    open("correlation_analysis_minatar.csv", "w"), fieldnames=["env", "draw", "dist"]
)
sample_lengths = []
printer.writeheader()
for env in sorted(
    [
        "Asterix-v1",
        "Breakout-v1",
        "Freeway-v1",
        "Seaquest-v1",
        "SpaceInvaders-v1",
    ]
):
    print(env)
    all_samples = []
    for model in [str(i) for i in range(1, 7)]:
        run = f"{env}-{model}"
        files = glob.glob(f"{runs[run]}/checkpoint/replay*")
        for files in tqdm.tqdm(files[:1:20]):
            try:
                data = np.load(files)["s"]
                all_samples.append(data)
            except Exception as e:
                print(e)
    all_samples = np.concatenate(all_samples, axis=1)
    all_samples = all_samples.reshape(-1, np.prod(all_samples.shape[-3:]))
    all_samples = strip_all_zero(all_samples)

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
