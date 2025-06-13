from typing import Callable, Dict

import jax
import jax.numpy as jnp
from aux_tasks.agents.actor_updates import get_actor_grads
from aux_tasks.agents.critic_updates import (
    get_critic_grads,
    get_discrete_critic_grads,
    soft_target_update,
)
from aux_tasks.agents.agent_config import AlgoHyperparams, Models
from aux_tasks.agents.model_updates import get_model_grads
from aux_tasks.rl_types import RLBatch


def jax_tree_norm(tree):
    return jnp.sqrt(
        jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_map(lambda x: jax.lax.stop_gradient(jnp.sum(x**2)), tree),
        )
    )


def full_update_step(
    model_batch: RLBatch,
    models: Models,
    hyperparams: AlgoHyperparams,
    loss_functions: Dict[str, Callable],
    key: jax.Array,
    batch_shape: Dict,
):
    critic_key, actor_key, model_key = jax.random.split(key, 3)

    # update model
    model_grads, model_loss_dict = get_model_grads(
        batch=model_batch,
        models=models,
        hyperparams=hyperparams,
        loss_functions=loss_functions,
        key=model_key,
        batch_shape=batch_shape,
    )

    # update critic
    critic_grads, critic_loss_dict = get_critic_grads(
        batch=model_batch,
        models=models,
        hyperparams=hyperparams,
    )
    actor_grads, actor_loss_dict = get_actor_grads(
        batch=model_batch,
        models=models,
        hyperparams=hyperparams,
        key=actor_key,
    )

    encoder_grads = model_grads.encoder.params

    grad_logs = {}

    grad_logs["model_encoder_grads"] = jax_tree_norm(encoder_grads)

    if not hyperparams.detach_critic_encoder:
        encoder_critic_grads = critic_grads["encoder"]
        encoder_grads = jax.tree_map(
            lambda x, y: x + y, encoder_grads, encoder_critic_grads
        )

        grad_logs["critic_encoder_grads"] = jax_tree_norm(encoder_critic_grads)

    if not hyperparams.detach_actor_encoder:
        encoder_actor_grads = actor_grads["encoder"]
        encoder_grads = jax.tree_map(
            lambda x, y: x + y, encoder_grads, encoder_actor_grads
        )

    new_encoder = models.encoder.apply_gradients(grads=encoder_grads)
    new_critic = models.critic.apply_gradients(grads=critic_grads["critic"])
    grad_logs["critic_grads"] = jax_tree_norm(critic_grads["critic"])
    new_actor = models.actor.apply_gradients(grads=actor_grads["actor"])
    grad_logs["actor_grads"] = jax_tree_norm(actor_grads["actor"])
    new_model = models.latent_model.apply_gradients(
        grads=model_grads.latent_model.params
    )
    grad_logs["model_grads"] = jax_tree_norm(model_grads.latent_model.params)
    new_decoder = models.decoder.apply_gradients(grads=model_grads.decoder.params)
    grad_logs["decoder_grads"] = jax_tree_norm(model_grads.decoder.params)

    new_encoder_target = soft_target_update(
        new_encoder, models.encoder_target, hyperparams.tau
    )
    new_critic_target = soft_target_update(
        new_critic, models.critic_target, hyperparams.tau
    )

    new_models = Models(
        critic=new_critic,
        critic_target=new_critic_target,
        actor=new_actor,
        encoder=new_encoder,
        encoder_target=new_encoder_target,
        latent_model=new_model,
        decoder=new_decoder,
    )

    return new_models, {
        "model": model_loss_dict,
        "critic": critic_loss_dict,
        "actor": actor_loss_dict,
        "grads": grad_logs,
    }


def dqn_update_step(
    model_batch: RLBatch,
    models: Models,
    hyperparams: AlgoHyperparams,
    loss_functions: Dict[str, Callable],
    key: jax.Array,
    batch_shape: Dict,
):
    critic_key, model_key = jax.random.split(key, 2)

    # update model
    model_grads, model_loss_dict = get_model_grads(
        batch=model_batch,
        models=models,
        hyperparams=hyperparams,
        loss_functions=loss_functions,
        key=model_key,
        batch_shape=batch_shape,
    )

    # update critic
    critic_grads, critic_loss_dict = get_discrete_critic_grads(
        batch=model_batch,
        models=models,
        hyperparams=hyperparams,
    )

    encoder_grads = model_grads.encoder.params

    grad_logs = {}

    grad_logs["model_encoder_grads"] = jax_tree_norm(encoder_grads)

    if not hyperparams.detach_critic_encoder:
        encoder_critic_grads = critic_grads["encoder"]
        encoder_grads = jax.tree_map(
            lambda x, y: x + y, encoder_grads, encoder_critic_grads
        )

        grad_logs["critic_encoder_grads"] = jax_tree_norm(encoder_critic_grads)

    new_encoder = models.encoder.apply_gradients(grads=encoder_grads)
    new_critic = models.critic.apply_gradients(grads=critic_grads["critic"])
    grad_logs["critic_grads"] = jax_tree_norm(critic_grads["critic"])
    new_model = models.latent_model.apply_gradients(
        grads=model_grads.latent_model.params
    )
    grad_logs["model_grads"] = jax_tree_norm(model_grads.latent_model.params)
    new_decoder = models.decoder.apply_gradients(grads=model_grads.decoder.params)
    grad_logs["decoder_grads"] = jax_tree_norm(model_grads.decoder.params)

    new_encoder_target = soft_target_update(
        new_encoder, models.encoder_target, hyperparams.tau
    )
    new_critic_target = soft_target_update(
        new_critic, models.critic_target, hyperparams.tau
    )

    new_models = Models(
        critic=new_critic,
        critic_target=new_critic_target,
        actor=models.actor,
        encoder=new_encoder,
        encoder_target=new_encoder_target,
        latent_model=new_model,
        decoder=new_decoder,
    )

    return new_models, {
        "model": model_loss_dict,
        "critic": critic_loss_dict,
        "grads": grad_logs,
    }
