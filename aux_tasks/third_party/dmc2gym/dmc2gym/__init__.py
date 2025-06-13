import sys
from typing import List

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register


def make(
    domain_name,
    task_name,
    seed=1,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None,
    time_limit=None,
    channels_first=True,
    action_noise=False,
    action_noise_type="normal",
    action_noise_level=0.0,
):
    env_id = "dmc_%s_%s_%s-v1" % (domain_name, task_name, seed)

    if from_pixels:
        assert (
            not visualize_reward
        ), "cannot use visualize reward when learning from pixels"

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if env_id not in gym.envs.registry.keys():
        task_kwargs = {}
        if seed is not None:
            task_kwargs["random"] = seed
        if time_limit is not None:
            task_kwargs["time_limit"] = time_limit
        register(
            id=env_id,
            entry_point="aux_tasks.third_party.dmc2gym.dmc2gym.wrappers:DMCWrapper",
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
                action_noise=action_noise,
                action_noise_type=action_noise_type,
                action_noise_level=action_noise_level,
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)


def vector_make(
    domain_name,
    task_name,
    num_envs,
    seeds,
    distraction: bool = False,
    noise_distraction: bool = False,
    obs_distortion: bool=False,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None,
    time_limit=None,
    channels_first=True,
    action_noise=False,
    action_noise_type="normal",
    action_noise_level=0.0,
):
    assert (
        len(seeds) == num_envs or len(seeds) == 1
    ), "seeds must be either of length 1 or equal to num_envs"
    if len(seeds) == 1:
        seeds = seeds * num_envs
    ids = []
    for i in range(num_envs):
        env_id = "dmc_%s_%s_%s-v1" % (domain_name, task_name, seeds[i])
        ids.append(env_id)

        if from_pixels:
            assert (
                not visualize_reward
            ), "cannot use visualize reward when learning from pixels"

        # shorten episode length
        max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

        if env_id not in gym.envs.registry.keys():
            task_kwargs = {}
            if seeds[i] is not None:
                task_kwargs["random"] = seeds[i]
            if time_limit is not None:
                task_kwargs["time_limit"] = time_limit
            register(
                id=env_id,
                entry_point="aux_tasks.third_party.dmc2gym.dmc2gym.wrappers:DMCWrapper",
                kwargs=dict(
                    domain_name=domain_name,
                    task_name=task_name,
                    task_kwargs=task_kwargs,
                    environment_kwargs=environment_kwargs,
                    visualize_reward=visualize_reward,
                    from_pixels=from_pixels,
                    height=height,
                    width=width,
                    camera_id=camera_id,
                    frame_skip=frame_skip,
                    channels_first=channels_first,
                    action_noise=action_noise,
                    action_noise_type=action_noise_type,
                    action_noise_level=action_noise_level,
                ),
                max_episode_steps=max_episode_steps,
            )
    envs = [gym.make(id) for id in ids]
    if distraction:
        print("Creating a distraction env")
        return DistractedDMCSequentialVectorEnv(envs, num_distractions=1, noise=noise_distraction, seed=seeds, frame_skip=2)
    return SequentialVectorEnv(envs, obs_distortion=obs_distortion, seed=seeds)


# gym.vector.AsyncVectorEnv([lambda: gym.make(id) for id in ids])
class SequentialVectorEnv:
    def __init__(self, envs, obs_distortion=False, seed=0):
        np.random.seed(seed)

        self.envs = envs
        self.action_space = envs[0].action_space
        self.observation_space = envs[0].observation_space
        self.dones = [True] * len(envs)
        self.last_obs = [0.0] * len(envs)

        self.obs_distortion = obs_distortion

        self.distortion_matrix = np.array([np.eye(
            int(np.prod(self.observation_space.shape).item())
        ) for _ in envs])

        if self.obs_distortion:
            for i, _ in enumerate(envs):
                invertible = False
                x = self.distortion_matrix[i]
                while not invertible:
                    x = np.random.choice(
                    [0., 1.], size=x.shape, p=[0.8, 0.2]
                    )
                    invertible = np.linalg.cond(x) < 1 / sys.float_info.epsilon
                    invertible = invertible and np.all(np.sum(x, axis=1) <= 255)
                self.distortion_matrix[i] = x

    def apply_distortion(self, obs, idx):
        if self.obs_distortion:
            obs_shape = obs.shape
            obs = np.dot(self.distortion_matrix[idx], obs.reshape(-1)).reshape(obs_shape)
        return obs

    def step(self, action):
        obs = []
        rew = []
        tru = []
        ter = []
        inf = []
        for idx, (act, env) in enumerate(zip(action, self.envs)):
            o, r, tr, te, i = env.step(act)
            obs.append(self.apply_distortion(o, idx))
            rew.append(r)
            tru.append(tr)
            ter.append(te)
            inf.append(i)
            self.dones[idx] = te or tr
            self.last_obs[idx] = o
        return (
            np.stack(obs, axis=0),
            np.stack(rew, axis=0),
            np.stack(tru, axis=0),
            np.stack(ter, axis=0),
            inf,
        )

    def reset(self, seed):
        info = []
        for idx, (env, s, d) in enumerate(zip(self.envs, seed, self.dones)):
            if d:
                o, i = env.reset(seed=s)
                info.append(i)
                self.last_obs[idx] = self.apply_distortion(o, idx)
        self.dones = [False] * len(self.envs)
        obs = np.stack(self.last_obs, axis=0)
        return obs, info


class DistractedSequentialVectorEnv:
    def __init__(self, envs, num_distractions=1, noise=False):
        self.envs = envs
        self.distraction_envs = [
            [gym.make("MinAtar/Freeway-v1") for _ in range(num_distractions)]
            for _ in envs
        ]
        self.action_space = envs[0].action_space
        self.observation_space = envs[0].observation_space
        # get obs_space
        obs, _ = envs[0].reset(seed=0)
        distr_obs = [env.reset()[0] for env in self.distraction_envs[0]]
        obs = np.concatenate((obs, np.concatenate(distr_obs, axis=-1)), axis=-1)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=obs.shape,
            dtype=obs.dtype,
        )
        print(self.observation_space.shape)
        self.dones = [True] * len(envs)
        self.last_obs = [0.0] * len(envs)
        self.noise = noise

    def step(self, action):
        obs = []
        rew = []
        tru = []
        ter = []
        inf = []
        for idx, (act, env) in enumerate(zip(action, self.envs)):
            o, r, tr, te, i = env.step(act)
            obs.append(o)
            rew.append(r)
            tru.append(tr)
            ter.append(te)
            inf.append(i)
            self.dones[idx] = te or tr

        # step_the_distraction_envs
        distr_obs = []
        for envs_list in self.distraction_envs:
            obses = []
            for env in envs_list:
                o, _, tr, te, _ = env.step(env.action_space.sample())
                if te or tr:
                    o, _ = env.reset()
                if self.noise:
                    o = np.random.choice([False, True], size=o.shape, p=[0.9, 0.1])
                obses.append(o)
            obses = np.concatenate(obses, axis=-1)
            distr_obs.append(obses)
        distr_obs = np.stack(distr_obs, axis=0)

        self.last_obs = np.concatenate((np.stack(obs, axis=0), distr_obs), axis=-1)

        return (
            np.concatenate((np.stack(obs, axis=0), distr_obs), axis=-1),
            np.stack(rew, axis=0),
            np.stack(tru, axis=0),
            np.stack(ter, axis=0),
            inf,
        )

    def reset(self, seed):
        info = []
        for idx, (env, s, d) in enumerate(zip(self.envs, seed, self.dones)):
            if d:
                o, i = env.reset(seed=s)
                info.append(i)
                distr_os = []
                for env in self.distraction_envs[idx]:
                    distr_os.append(env.reset()[0])
                o = np.concatenate((o, np.concatenate(distr_os, axis=-1)), axis=-1)
                self.last_obs[idx] = o
        self.dones = [False] * len(self.envs)
        obs = np.stack(self.last_obs, axis=0)
        return obs, info


class DistractedDMCSequentialVectorEnv:
    def __init__(self, envs, num_distractions=1, noise=False, seed: List=[0], frame_skip=2):
        print(seed)
        self.envs = envs
        self.num_distractions = num_distractions
        self.distraction_envs = [
            vector_make(
                "humanoid",
                "run",
                num_distractions,
                seeds=[seed[i]],
                frame_skip=frame_skip,
                episode_length=1000,
            ) for i, _ in enumerate(envs)
        ]
        self.action_space = envs[0].action_space
        self.observation_space = envs[0].observation_space
        # get obs_space
        obs, _ = envs[0].reset(seed=0)
        distr_obs = self.distraction_envs[0].reset(seed=seed)[0]
        distr_obs = distr_obs.reshape(-1)
        print(obs.shape)
        print(distr_obs.shape)
        obs = np.concatenate((obs, distr_obs), axis=-1)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=obs.dtype,
        )
        print(self.observation_space.shape)
        self.dones = [True] * len(envs)
        self.last_obs = [0.0] * len(envs)
        self.noise = noise

    def step(self, action):
        obs = []
        rew = []
        tru = []
        ter = []
        inf = []
        for idx, (act, env) in enumerate(zip(action, self.envs)):
            o, r, tr, te, i = env.step(act)
            obs.append(o)
            rew.append(r)
            tru.append(tr)
            ter.append(te)
            inf.append(i)
            self.dones[idx] = te or tr

        # step_the_distraction_envs
        distr_obses = []
        for env in self.distraction_envs:
            action = []
            for _ in range(self.num_distractions):
                action.append(env.action_space.sample())
            o, _, tr, te, _ = env.step(np.stack(action, axis=0))
            if te or tr:
                o, _ = env.reset([0]* self.num_distractions)
            if self.noise:
                o = np.random.randn(*o.shape)
            o = o.reshape(-1)
            distr_obses.append(o)
        distr_obs = np.stack(distr_obses, axis=0)

        self.last_obs = np.concatenate((np.stack(obs, axis=0), distr_obs), axis=-1)

        return (
            self.last_obs,
            np.stack(rew, axis=0),
            np.stack(tru, axis=0),
            np.stack(ter, axis=0),
            inf,
        )

    def reset(self, seed):
        info = []
        for idx, (env, s, d) in enumerate(zip(self.envs, seed, self.dones)):
            if d:
                o, i = env.reset(seed=s)
                info.append(i)
                distr_os = []
                distr_os.append(self.distraction_envs[idx].reset([seed[idx]])[0].reshape(-1))
                o = np.concatenate((o, np.concatenate(distr_os, axis=-1)), axis=-1)
                self.last_obs[idx] = o
        self.dones = [False] * len(self.envs)
        obs = np.stack(self.last_obs, axis=0)
        return obs, info
