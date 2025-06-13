from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from aux_tasks.agents.agent_config import AlgoHyperparams, Models

from aux_tasks.rl_types import RLBatch


def get_actor_grads(
    batch: RLBatch,
    models: Models,
    hyperparams: AlgoHyperparams,
    key: jax.Array,
) -> Tuple[Dict, Dict]:
    state = batch.state

    def _j(actor_params, encoder_params):
        latent_state = models.encoder.apply_fn(encoder_params, state)

        action = models.actor.apply_fn(actor_params, latent_state)
        q1, q2 = models.critic.apply_fn(
            models.critic.params, jax.lax.stop_gradient(latent_state), action
        )
        q = jnp.minimum(q1, q2)

        return -jnp.mean(q), {
            "mean_action": jnp.mean(action),
            "mean_abs_action": jnp.mean(jnp.abs(action)),
        }

    loss_grad = jax.value_and_grad(_j, argnums=[0, 1], has_aux=True, allow_int=True)
    (target, loss_dict), grad = loss_grad(models.actor.params, models.encoder.params)
    return {"actor": grad[0], "encoder": grad[1]}, {"actor_loss": target, **loss_dict}
