from typing import Dict
import jax
from flax.core.frozen_dict import FrozenDict

from aux_tasks.rl_types import RLBatch
from aux_tasks.model_utils.loss_functions import compute_model_loss
from aux_tasks.agents.agent_config import (
    AlgoHyperparams,
    Models,
)
from aux_tasks.utils.jax import batch_loss_fn


def get_model_grads(
    batch: RLBatch,
    models: Models,
    hyperparams: AlgoHyperparams,
    loss_functions: FrozenDict,
    key: jax.Array,
    batch_shape: Dict,
):
    batched_loss_function = batch_loss_fn(
        compute_model_loss,
        in_axes=(
            batch_shape,
            None,
            0,
            None,
            None,
        ),
        out_axes=(0, 0),
        has_aux=True,
    )
    loss_grad = jax.value_and_grad(
        batched_loss_function,
        argnums=1,
        has_aux=True,
        allow_int=True,
    )
    keys = jax.random.split(key, batch.state.shape[0])
    ((_, loss_dict), grads) = loss_grad(
        batch,
        models,
        keys,
        hyperparams,
        loss_functions,
    )

    # log encoder statistics
    latent = models.encoder.apply_fn(models.encoder.params, batch.state)
    latent_std = latent.std(axis=0).mean()
    latent_mean = latent.mean()
    encoder_statistics = {
        "latent_std": latent_std,
        "latent_mean": latent_mean,
    }

    return grads, {**loss_dict, **encoder_statistics}
