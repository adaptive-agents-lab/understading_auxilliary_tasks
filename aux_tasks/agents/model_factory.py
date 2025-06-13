from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import jax
import optax
from flax.core import frozen_dict
from flax.training import train_state
from aux_tasks.agents.agent_config import Models

from aux_tasks.data.replay_buffer import ReplayBuffer
from aux_tasks.rl_types import (
    AbstractActor,
    AbstractCritic,
    AbstractDecoder,
    AbstractEncoder,
    AbstractLatentModel,
)
from aux_tasks.model_utils.loss_functions import (
    byol_crossent,
    byol_mse,
    cosine_similarity,
    reconstruction,
    reward_prediction,
)


@dataclass
class ModelFactory:
    critic: AbstractCritic
    critic_target: AbstractCritic
    actor: AbstractActor
    encoder: AbstractEncoder
    encoder_target: AbstractEncoder
    latent_model: AbstractLatentModel
    decoder: AbstractDecoder
    replay_buffer: ReplayBuffer
    encoder_learning_rate: float
    model_learning_rate: float
    critic_learning_rate: float
    actor_learning_rate: float
    gradient_clip: float
    seed: int

    def __post_init__(self):
        self.key = jax.random.PRNGKey(self.seed)
        self.key, init_key = jax.random.split(self.key)

        sample_batch = self.replay_buffer.get_dummy_batch()
        dummy_state = sample_batch.state
        dummy_action = sample_batch.action
        num_seeds = self.replay_buffer.num_seeds

        # initialize all models
        (
            critic_key,
            critic_target_key,
            actor_key,
            encoder_key,
            latent_model_key,
            decoder_key,
        ) = jax.random.split(init_key, 6)

        def _create_train_states(keys):
            # split the keys
            critic_key = keys[0]
            actor_key = keys[1]
            encoder_key = keys[2]
            latent_model_key = keys[3]
            decoder_key = keys[4]
            critic_target_key = keys[5]

            # encoder
            encoder_params = self.encoder.init(encoder_key, dummy_state)
            encoder_target_params = self.encoder_target.init(encoder_key, dummy_state)

            # get latent state
            dummy_latent_state = self.encoder.apply(encoder_params, dummy_state)

            # model
            latent_model_params = self.latent_model.init(
                latent_model_key, dummy_latent_state, dummy_action
            )

            # decoder
            decoder_params = self.decoder.init(
                decoder_key,
                dummy_latent_state,
            )

            # critic
            critic_params = self.critic.init(
                critic_key,
                dummy_latent_state,
                dummy_action,
            )
            target_params = self.critic_target.init(
                critic_target_key,
                dummy_latent_state,
                dummy_action,
            )

            # actor
            actor_params = self.actor.init(actor_key, dummy_latent_state)

            # initialize optimizers
            def _add_clip(optimizer):
                return optax.chain(
                    optax.clip_by_global_norm(self.gradient_clip), optimizer
                )

            critic_optim = _add_clip(
                optax.adam(learning_rate=self.critic_learning_rate)
            )
            actor_optim = _add_clip(optax.adam(learning_rate=self.actor_learning_rate))
            encoder_optim = _add_clip(
                optax.adam(learning_rate=self.encoder_learning_rate)
            )
            latent_model_optim = _add_clip(
                optax.adam(learning_rate=self.model_learning_rate)
            )
            decoder_model_optim = _add_clip(
                optax.adam(learning_rate=self.model_learning_rate)
            )

            # create train states
            critic_train_state = train_state.TrainState.create(
                apply_fn=self.critic.apply,
                params=critic_params,
                tx=critic_optim,
            )
            target_train_state = train_state.TrainState.create(
                apply_fn=self.critic.apply,
                params=target_params,
                tx=optax.set_to_zero(),
            )
            actor_train_state = train_state.TrainState.create(
                apply_fn=self.actor.apply, params=actor_params, tx=actor_optim
            )
            encoder_train_state = train_state.TrainState.create(
                apply_fn=self.encoder.apply, params=encoder_params, tx=encoder_optim
            )
            encoder_target_train_state = train_state.TrainState.create(
                apply_fn=self.encoder_target.apply,
                params=encoder_target_params,
                tx=optax.set_to_zero(),
            )
            latent_model_train_state = train_state.TrainState.create(
                apply_fn=self.latent_model.apply,
                params=latent_model_params,
                tx=latent_model_optim,
            )
            decoder_train_state = train_state.TrainState.create(
                apply_fn=self.decoder.apply,
                params=decoder_params,
                tx=decoder_model_optim,
            )

            return (
                critic_train_state,
                target_train_state,
                actor_train_state,
                encoder_train_state,
                encoder_target_train_state,
                latent_model_train_state,
                decoder_train_state,
            )

        if num_seeds > 1:
            critic_key = jax.random.split(critic_key, num_seeds)
            actor_key = jax.random.split(actor_key, num_seeds)
            encoder_key = jax.random.split(encoder_key, num_seeds)
            latent_model_key = jax.random.split(latent_model_key, num_seeds)
            decoder_key = jax.random.split(decoder_key, num_seeds)
            critic_target_key = jax.random.split(critic_target_key, num_seeds)

            (
                self.critic_state,
                self.target_state,
                self.actor_state,
                self.encoder_state,
                self.encoder_target_state,
                self.latent_model_state,
                self.decoder_state,
            ) = jax.vmap(_create_train_states)(
                (
                    critic_key,
                    critic_target_key,
                    actor_key,
                    encoder_key,
                    latent_model_key,
                    decoder_key,
                )
            )
        else:
            (
                self.critic_state,
                self.target_state,
                self.actor_state,
                self.encoder_state,
                self.encoder_target_state,
                self.latent_model_state,
                self.decoder_state,
            ) = _create_train_states(
                (
                    critic_key,
                    critic_target_key,
                    actor_key,
                    encoder_key,
                    latent_model_key,
                    decoder_key,
                )
            )

    def get_networks(self) -> Models:
        return Models(
            critic=self.critic_state,
            critic_target=self.target_state,
            actor=self.actor_state,
            encoder=self.encoder_state,
            encoder_target=self.encoder_target_state,
            latent_model=self.latent_model_state,
            decoder=self.decoder_state,
        )

    def init(self) -> Models:
        return self.get_networks()


def w_fn(w: float, fn: Callable):
    return lambda *args: w * fn(*args)


def make_loss_functions(loss_functions: Sequence[Tuple[str, float]]):
    loss_fn_dict = {}
    for fn, w in loss_functions:
        match fn:
            case "crossent" | "crossentropy" | "byol_crossent" | "byol_crossentropy":
                loss_fn_dict["byol_crossent"] = w_fn(w, byol_crossent)
            case "byol_mse" | "mse":
                loss_fn_dict["byol_mse"] = w_fn(w, byol_mse)
            case "cosine" | "cosine_sim":
                loss_fn_dict["cosine_sim"] = w_fn(w, cosine_similarity)
            case "reward_recon":
                loss_fn_dict["reward_recon"] = w_fn(w, reward_prediction)
            case "recon" | "reconstruction" | "obs_recon" | "obs":
                loss_fn_dict["recon_obs"] = w_fn(w, reconstruction)
            case _:
                raise ValueError(f"Invalid loss function: {fn}")
    return frozen_dict.freeze(loss_fn_dict)
