import gymnasium as gym
import numpy as np
import tqdm

from aux_tasks.third_party.dmc2gym.dmc2gym.sequential import (
    DistractedSequentialVectorEnv,
)

env = [gym.make("MinAtar/Asterix-v1") for i in range(10)]

env = DistractedSequentialVectorEnv(env, num_distractions=1, noise=False)

env.reset(seed=[0] * 10)

print(env.observation_space)

exit()

cum_rew = []
ep_rews = []
for i in tqdm.tqdm(range(int(5e6) * 5)):
    obs, rew, ter, trunc, info = env.step([0 for _ in range(10)])
    cum_rew.append(rew)
    if np.any(ter) or np.any(trunc):
        ep_rews.append(sum(cum_rew))
        cum_rew = []
        env.reset([0] * 10)
print(np.mean(ep_rews))
