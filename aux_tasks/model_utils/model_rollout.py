from typing import Dict

import jax
from jax import numpy as jnp
from flax.training.train_state import TrainState


def generate_latent_rollout_from_actions(
    start_state: jax.Array,
    actions: jax.Array,
    latent_model: TrainState,
    decoder: TrainState,
) -> Dict[str, jax.Array]:
    """Generate a latent rollout."""

    def _predict_next(
        state: jax.Array,
        action: jax.Array,
    ):
        next_latent, next_reward = latent_model.apply_fn(
            latent_model.params, state, action
        )
        recon = decoder.apply_fn(decoder.params, next_latent)
        return next_latent, {
            "action": action,
            "reward": next_reward,
            "latent": next_latent,
            "recon": recon,
        }

    _, return_dict = jax.lax.scan(_predict_next, start_state, actions)  # type: ignore
    return_dict["latent"] = jnp.concatenate(
        [jax.lax.expand_dims(start_state, dimensions=[0]), return_dict["latent"]]
    )
    first_recon = decoder.apply_fn(decoder.params, start_state)
    return_dict["recon"] = jnp.concatenate(
        [jax.lax.expand_dims(first_recon, dimensions=[0]), return_dict["recon"]]
    )
    return return_dict
