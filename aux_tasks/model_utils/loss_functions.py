from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp


from aux_tasks.model_utils.model_rollout import generate_latent_rollout_from_actions
from aux_tasks.agents.agent_config import AlgoHyperparams, Models
from aux_tasks.rl_types import RLBatch


def byol_crossent(
    obs: jax.Array,
    pred_obs: jax.Array,
    z: jax.Array,
    pred_z: jax.Array,
    rewards: jax.Array,
    pred_rewards: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    return -jnp.mean(weight[-1] * jax.lax.stop_gradient(z) * jnp.log(pred_z))


def byol_mse(
    obs: jax.Array,
    pred_obs: jax.Array,
    z: jax.Array,
    pred_z: jax.Array,
    rewards: jax.Array,
    pred_rewards: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    return jnp.mean(weight[-1] * (jax.lax.stop_gradient(z) - pred_z) ** 2)


def cosine_similarity(
    obs: jax.Array,
    pred_obs: jax.Array,
    z: jax.Array,
    pred_z: jax.Array,
    rewards: jax.Array,
    pred_rewards: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    pred = jax.lax.stop_gradient(jax.lax.normalize(z, axis=-1))
    target = jax.lax.normalize(pred_z, axis=-1)
    return jnp.mean(weight[-1] * (pred - target) ** 2)


def reward_prediction(
    obs: jax.Array,
    pred_obs: jax.Array,
    z: jax.Array,
    pred_z: jax.Array,
    rewards: jax.Array,
    pred_rewards: jax.Array,
    weight: jax.Array,
):
    return jnp.mean(weight[-1] * (rewards - pred_rewards) ** 2)


def reconstruction(
    obs: jax.Array,
    pred_obs: jax.Array,
    z: jax.Array,
    pred_z: jax.Array,
    rewards: jax.Array,
    pred_rewards: jax.Array,
    weight: jax.Array,
):
    return jnp.mean(weight[-1] * (obs - pred_obs) ** 2)


def reconstruction_crossent(
    obs: jax.Array,
    pred_obs: jax.Array,
    z: jax.Array,
    pred_z: jax.Array,
    rewards: jax.Array,
    pred_rewards: jax.Array,
    weight: jax.Array,
):
    return -jnp.mean(weight[-1] * obs * jnp.log(pred_obs))


def compute_model_loss(
    batch: RLBatch,
    models: Models,
    hyperparams: AlgoHyperparams,
    key: jax.Array,
    loss_functions: Dict[str, Callable],
) -> Tuple[jax.Array, Dict]:
    # get encoder outputs
    encoder = models.encoder
    encoder_target = models.encoder_target
    decoder = models.decoder
    model = models.latent_model

    state = batch.state
    actions = batch.action
    reward = batch.reward
    next_states = batch.next_state
    mask = batch.mask

    all_states = jnp.concatenate(
        [jax.lax.expand_dims(state, dimensions=[0]), next_states], axis=0
    )

    latent_state: jax.Array = encoder.apply_fn(encoder.params, state)  # type: ignore
    latent_states: jax.Array = jax.lax.stop_gradient(encoder_target.apply_fn(encoder_target.params, all_states))  # type: ignore

    # get a model rollout
    model_rollout = generate_latent_rollout_from_actions(
        latent_state,
        actions,
        model,
        decoder,
    )
    predicted_latents = model_rollout["latent"]
    predicted_reconstructions = model_rollout["recon"]
    predicted_rewards = model_rollout["reward"]

    losses = {}
    total_loss = 0.0

    for loss_name, loss_f in loss_functions.items():
        loss = loss_f(
            all_states,
            predicted_reconstructions,
            latent_states,
            predicted_latents,
            reward,
            predicted_rewards,
            mask,
        )
        loss = jnp.mean(loss)
        losses[loss_name] = loss
        total_loss += loss
        # jax.debug.print("{}".format(loss_name) + ": {}", loss)
    return total_loss, {"total_loss": total_loss, **losses}
