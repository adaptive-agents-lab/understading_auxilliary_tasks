from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from aux_tasks.rl_types import RLBatch
from aux_tasks.agents.agent_config import (
    Models,
    AlgoHyperparams,
)


def compute_td_target_from_state(
    state: jax.Array,
    reward: jax.Array,
    critic: TrainState,
    actor: TrainState,
    gamma: float,
):
    action = actor.apply_fn(actor.params, state)
    value = critic.apply_fn(critic.params, state, action)
    value = jnp.stack(value, axis=-1)
    value = value.min(axis=-1)
    td_target = reward + gamma * value
    return td_target


def soft_target_update(
    source: TrainState, target: TrainState, tau: float
) -> TrainState:
    source_params = source.params
    target_params = target.params
    new_target_params = jax.tree_util.tree_map(
        lambda x, y: tau * x + (1 - tau) * y, target_params, source_params
    )

    new_target = TrainState.create(
        apply_fn=target.apply_fn,
        params=new_target_params,
        tx=target.tx,
    )
    return new_target


def get_critic_grads(
    batch: RLBatch,
    models: Models,
    hyperparams: AlgoHyperparams,
) -> Tuple[Models, Dict]:
    state = batch.state
    action = batch.action[:, 0]
    reward = batch.reward[:, 0]
    next_state = batch.next_state[:, 0]
    mask = batch.mask[:, 0]

    def _j(critic_params, encoder_params):

        latent_next_state = models.encoder_target.apply_fn(
            models.encoder_target.params, next_state
        )
        next_action = models.actor.apply_fn(models.actor.params, latent_next_state)
        q1, q2 = models.critic.apply_fn(
            models.critic_target.params, latent_next_state, next_action
        )
        q = jnp.minimum(q1, q2)
        target = reward + hyperparams.gamma * q
        target = jax.lax.stop_gradient(target)

        latent_state = models.encoder.apply_fn(encoder_params, state)
        q1, q2 = models.critic.apply_fn(critic_params, latent_state, action)
        critic_loss = jnp.mean(mask * (q1 - target) ** 2) + jnp.mean(
            mask * (q2 - target) ** 2
        )

        return (
            critic_loss,
            {  # type: ignore
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
                "q1_max": q1.max(),
                "q2_max": q2.max(),
            },
        )

    loss_func = jax.value_and_grad(_j, argnums=[0, 1], has_aux=True, allow_int=True)
    (_, loss_dict), grad = loss_func(models.critic.params, models.encoder.params)

    return {"critic": grad[0], "encoder": grad[1]}, loss_dict


def get_discrete_critic_grads(
    batch: RLBatch,
    models: Models,
    hyperparams: AlgoHyperparams,
) -> Tuple[Models, Dict]:
    state = batch.state
    action = batch.action[:, 0]
    reward = batch.reward[:, 0]
    next_state = batch.next_state[:, 0]
    mask = batch.mask[:, 0]

    def _j(critic_params, encoder_params):

        latent_next_state = models.encoder_target.apply_fn(
            models.encoder_target.params, next_state
        )
        next_q = jnp.take_along_axis(
            models.critic.apply_fn(models.critic_target.params, latent_next_state),
            jnp.argmax(
                models.critic.apply_fn(models.critic.params, latent_next_state),
                axis=-1,
                keepdims=True,
            ),
            axis=-1,
        )

        target = reward + mask * hyperparams.gamma * next_q
        target = jax.lax.stop_gradient(target)

        # latent_state = models.encoder.apply_fn(encoder_params, state)
        q = models.critic.apply_fn(
            critic_params, models.encoder.apply_fn(encoder_params, state)
        )
        # I implemented action to be a one-hot vector, which works weirdly here
        # all other qs are zeroed,
        # max_action = jnp.argmax(action, axis=-1, keepdims=True)
        q = jnp.take_along_axis(q, jnp.argmax(action, axis=-1, keepdims=True), axis=-1)
        critic_loss = jnp.mean((q - target) ** 2)

        return (
            critic_loss,
            {  # type: ignore
                "critic_loss": critic_loss,
                "q": q.mean(),
                "q_max": q.max(),
            },
        )

    loss_func = jax.value_and_grad(_j, argnums=[0, 1], has_aux=True, allow_int=True)
    (_, loss_dict), grad = loss_func(models.critic.params, models.encoder.params)

    return {"critic": grad[0], "encoder": grad[1]}, loss_dict
