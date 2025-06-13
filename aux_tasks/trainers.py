from dataclasses import dataclass
import os

import numpy as onp
import jax
import jax.numpy as jnp
from tqdm import tqdm
from aux_tasks.agents.critic_updates import soft_target_update

from aux_tasks.agents.agent_config import AlgoHyperparams, Models

from aux_tasks.rl_types import (
    AbstractActor,
    AbstractCritic,
    AbstractDecoder,
    AbstractEncoder,
    AbstractLatentModel,
)
from aux_tasks.data.env import Env, MinAtarEnv
from aux_tasks.data.replay_buffer import ReplayBuffer
from aux_tasks.utils.checkpointing import CheckpointHandler
from aux_tasks.utils.jax import tree_list_mean
from aux_tasks.agents.model_factory import (
    ModelFactory,
    make_loss_functions,
)
from aux_tasks.agents.update_function import dqn_update_step, full_update_step
from aux_tasks.utils.logging import (
    Logger,
    multi_seed_return_dict,
)


@dataclass
class TrainHyperparams:
    seed: int
    num_seeds: int
    init_steps: int
    env_steps: int
    update_steps: int
    total_steps: int
    full_model_length_steps: int
    full_model_use_steps: int
    model_batch_size: int
    rl_batch_size: int
    action_samples: int
    log_freq: int
    save_freq: int
    eval_freq: int
    eval_episodes: int
    save_path: str
    hard_target_update: int


class SingleSeedTrainer:
    def __init__(
        self,
        critic: AbstractCritic,
        critic_target: AbstractCritic,
        actor: AbstractActor,
        encoder: AbstractEncoder,
        encoder_target: AbstractEncoder,
        latent_model: AbstractLatentModel,
        decoder: AbstractDecoder,
        replay_buffer: ReplayBuffer,
        env: Env,
        algo_hyperparams: AlgoHyperparams,
        train_hyperparams: TrainHyperparams,
        logger: Logger,
    ):
        models = ModelFactory(
            critic=critic,
            critic_target=critic_target,
            actor=actor,
            encoder=encoder,
            encoder_target=encoder_target,
            latent_model=latent_model,
            decoder=decoder,
            replay_buffer=replay_buffer,
            encoder_learning_rate=algo_hyperparams.encoder_learning_rate,
            model_learning_rate=algo_hyperparams.model_learning_rate,
            critic_learning_rate=algo_hyperparams.critic_learning_rate,
            actor_learning_rate=algo_hyperparams.actor_learning_rate,
            gradient_clip=algo_hyperparams.gradient_clip,
            seed=train_hyperparams.seed,
        )
        self.models = models.init()

        self.encoder = encoder
        self.latent_model = latent_model
        self.critic = critic
        self.actor = actor

        # setup replay buffer
        self.replay_buffer = replay_buffer

        # setup shapes for vmap
        self.batch_shape = jax.tree_map(
            lambda x: 0, self.replay_buffer.get_dummy_batch()
        )

        # setup hyperparams
        self.algo_hyperparams = algo_hyperparams
        self.train_hyperparams = train_hyperparams
        self.loss_functions = make_loss_functions(algo_hyperparams.loss_functions)

        # setup utility
        self.key = jax.random.PRNGKey(train_hyperparams.seed)
        self.key, env_step_key = jax.random.split(self.key)

        # setup env
        self.env = env
        self.env_reset = self.env.get_n_reset()
        self.env_step = self.env.get_n_step(env_step_key)

        self.logger = logger

        self.checkpointer = CheckpointHandler(train_hyperparams.save_path)

        # handle reloading logic here
        self.steps_done = 0

    def check_pretrain(self, path, alt_path):
        if alt_path:
            path = os.path.join(alt_path, path)

        if os.path.exists(os.path.join(path, "steps_done.txt")):
            self.load(path)
            with open(os.path.join(path, "steps_done.txt"), "r") as f:
                self.steps_done = int(f.read())
            print(f"Resuming from previous checkpoint at {self.steps_done}")
        else:
            print("No checkpoint found, starting from scratch")

    def save(self, path):
        self.replay_buffer.save(path)
        self.checkpointer.checkpoint_params(self.models, self.steps_done)
        with open(os.path.join(path, "steps_done.txt"), "w") as f:
            f.write(str(self.steps_done))

    def load(self, path):
        self.replay_buffer.load(path)
        self.models = self.checkpointer.restore_params(self.models, path)


class MultiSeedTrainer(SingleSeedTrainer):
    def __init__(
        self,
        critic: AbstractCritic,
        critic_target: AbstractCritic,
        actor: AbstractActor,
        encoder: AbstractEncoder,
        encoder_target: AbstractEncoder,
        latent_model: AbstractLatentModel,
        decoder: AbstractDecoder,
        replay_buffer: ReplayBuffer,
        env: Env,
        algo_hyperparams: AlgoHyperparams,
        train_hyperparams: TrainHyperparams,
        logger: Logger,
    ):
        # setup models
        models = ModelFactory(
            critic=critic,
            critic_target=critic_target,
            actor=actor,
            encoder=encoder,
            encoder_target=encoder_target,
            latent_model=latent_model,
            decoder=decoder,
            replay_buffer=replay_buffer,
            encoder_learning_rate=algo_hyperparams.encoder_learning_rate,
            model_learning_rate=algo_hyperparams.model_learning_rate,
            critic_learning_rate=algo_hyperparams.critic_learning_rate,
            actor_learning_rate=algo_hyperparams.actor_learning_rate,
            gradient_clip=algo_hyperparams.gradient_clip,
            seed=train_hyperparams.seed,
        )
        self.models = models.init()

        self.encoder = encoder
        self.latent_model = latent_model
        self.critic = critic
        self.actor = actor

        # setup replay buffer
        self.replay_buffer = replay_buffer

        # setup shapes for vmap
        self.batch_shape = jax.tree_map(
            lambda x: 0, self.replay_buffer.get_dummy_batch()
        )

        # setup env
        self.env = env
        self.env_reset = self.env.get_n_reset()
        self.env_step = self.env.get_n_step()

        # setup hyperparams
        self.algo_hyperparams = algo_hyperparams
        self.train_hyperparams = train_hyperparams
        self.loss_functions = make_loss_functions(algo_hyperparams.loss_functions)

        # setup utility
        self.key = jax.random.PRNGKey(train_hyperparams.seed)
        # extra split because of testing
        # self.key = jax.random.split(self.key, 3)[0]
        self.logger = logger

        self.checkpointer = CheckpointHandler(train_hyperparams.save_path)
        # handle reloading logic here
        self.steps_done = 0

    def train(self):
        num_seeds = self.train_hyperparams.num_seeds
        key, reset_key = jax.random.split(self.key)
        done = False
        rewards = 0.0
        ep_rewards = [[0.0] * 10 for _ in range(num_seeds)]
        reset_rng = jax.random.split(reset_key, num_seeds)
        state = self.env_reset(reset_rng)

        return_dicts = []

        total_steps = self.train_hyperparams.total_steps

        if isinstance(self.env, MinAtarEnv):
            vmaped_action = jax.jit(
                jax.vmap(
                    lambda s, m, random, key: get_critic_action(
                        s,
                        m,
                        0.05,
                        random,
                        key,
                    ),
                    in_axes=(
                        0,
                        0,
                        None,
                        0,
                    ),
                ),
                static_argnames=("random"),
            )
            vmaped_update = jax.jit(
                jax.vmap(
                    dqn_update_step,
                    in_axes=(self.batch_shape, 0, None, None, 0, None),
                ),
                static_argnames=("hyperparams", "loss_functions", "batch_shape"),
            )
        else:
            vmaped_action = jax.jit(
                jax.vmap(
                    lambda s, m, random, key: get_policy_action(
                        s,
                        m,
                        0.1,
                        random,
                        key,
                    ),
                    in_axes=(
                        0,
                        0,
                        None,
                        0,
                    ),
                ),
                static_argnames=("random"),
            )
            vmaped_update = jax.jit(
                jax.vmap(
                    full_update_step,
                    in_axes=(self.batch_shape, 0, None, None, 0, None),
                ),
                static_argnames=("hyperparams", "loss_functions", "batch_shape"),
            )

        with tqdm(total=total_steps) as pbar:
            # pbar.update(self.steps_done)
            while self.steps_done <= total_steps:
                for _ in range(self.train_hyperparams.env_steps):
                    # key logic
                    key, step_key = jax.random.split(key)
                    step_keys = jax.random.split(step_key, num_seeds)
                    key, action_key = jax.random.split(key)
                    action_keys = jax.random.split(action_key, num_seeds)
                    action = vmaped_action(
                        state.obs,
                        self.models,
                        self.steps_done < self.train_hyperparams.init_steps,
                        action_keys,
                    )
                    next_state, reward, done = self.env_step(
                        step_keys, state.state, action
                    )
                    rewards += onp.array(reward)
                    self.replay_buffer.insert(
                        state.obs, action, reward, done, next_state.obs
                    )
                    if jax.numpy.any(done):
                        reset_key, reset_rng = jax.random.split(reset_key)
                        reset_rng = jax.random.split(reset_rng, num_seeds)
                        for i, d in enumerate(done):
                            if d:
                                ep_rewards[i] = ep_rewards[i][1:] + [rewards[i].item()]
                                rewards[i] = 0.0
                        next_state = self.env_reset(reset_rng)
                    state = next_state
                # pbar.update(self.train_hyperparams.env_steps)
                self.steps_done += self.train_hyperparams.env_steps
                if self.steps_done > self.train_hyperparams.init_steps:
                    if self.steps_done % self.train_hyperparams.hard_target_update == 0:
                        self.models = jax.jit(jax.vmap(vmapped_soft_update))(
                            self.models
                        )
                    for _ in range(self.train_hyperparams.update_steps):
                        # training

                        # split all required keys
                        key, train_subkey = jax.random.split(key)
                        model_key, critic_key, actor_key = jax.random.split(
                            train_subkey, 3
                        )
                        key, train_step_key = jax.random.split(key)
                        train_step_key = jax.random.split(train_step_key, num_seeds)

                        key, mb_key = jax.random.split(key)

                        # sample batches
                        model_batch = self.replay_buffer.sample(
                            self.train_hyperparams.model_batch_size, model_key  # type: ignore
                        )

                        # core update function (jitted for fixed hyperparams)
                        self.models, return_dict = vmaped_update(
                            model_batch,
                            self.models,
                            self.algo_hyperparams,
                            self.loss_functions,
                            train_step_key,
                            self.batch_shape,
                        )
                        return_dicts.append(return_dict)

                    # logging

                    if self.steps_done % self.train_hyperparams.log_freq == 0:
                        return_dict = tree_list_mean(return_dicts)
                        return_dict = multi_seed_return_dict(
                            return_dict, self.train_hyperparams.num_seeds
                        )
                        ep_rew_dict = {"ep_rew": jnp.array(ep_rewards).mean(axis=1)}
                        ep_rew_dict = multi_seed_return_dict(ep_rew_dict, num_seeds)
                        self.logger.log({**return_dict, **ep_rew_dict}, self.steps_done)
                        del return_dicts
                        return_dicts = []

                    if self.steps_done % self.train_hyperparams.save_freq == 0:
                        # assert jax.numpy.all(
                        #     done
                        # ), "Checkpointing is only done after a done to ensure consistent data logging"
                        self.save(self.train_hyperparams.save_path)
                        self.logger.flush()


def get_policy_action(
    state: jax.Array,
    models: Models,
    exploration_noise_sigma: jax.Array,
    random: bool,
    rand_key: jax.Array,
):
    feature = models.encoder.apply_fn(models.encoder.params, state)
    action = models.actor.apply_fn(models.actor.params, feature)

    # exploration noise from drq v2 paper
    exploration_noise = (
        jax.random.normal(rand_key, shape=action.shape) * exploration_noise_sigma
    )
    exploration_noise = jax.numpy.clip(exploration_noise, -0.3, 0.3)
    action += exploration_noise
    action = jax.numpy.clip(action, -1.0, 1.0)

    if random:
        action = jax.random.uniform(rand_key, shape=action.shape, minval=-1, maxval=1)
    return action


def get_critic_action(
    state: jax.Array,
    models: Models,
    exploration_noise_sigma: jax.Array,
    random: bool,
    rand_key: jax.Array,
):
    explore = jax.random.uniform(rand_key) < exploration_noise_sigma
    feature = models.encoder.apply_fn(models.encoder.params, state)
    qs = models.critic.apply_fn(models.critic.params, feature)
    action = jax.numpy.argmax(qs, axis=-1)
    random_qs = jax.random.uniform(rand_key, shape=qs.shape, minval=-1, maxval=1)
    random_action = jax.numpy.argmax(random_qs, axis=-1)

    action = jnp.where(explore, random_action, action)

    if random:
        action = random_action
    return action


def vmapped_soft_update(models):
    return Models(
        critic=models.critic,
        critic_target=soft_target_update(models.critic_target, models.critic, 1.0),
        actor=models.actor,
        encoder=models.encoder,
        encoder_target=soft_target_update(models.encoder_target, models.encoder, 1.0),
        latent_model=models.latent_model,
        decoder=models.decoder,
    )
