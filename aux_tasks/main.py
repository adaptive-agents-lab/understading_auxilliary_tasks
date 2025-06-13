import os
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError

from jax import config

from aux_tasks.agents.agent_config import AlgoHyperparams
from aux_tasks.trainers import (
    MultiSeedTrainer,
    SingleSeedTrainer,
    TrainHyperparams,
)
from aux_tasks.data.env import Env, EnvConfig, MinAtarEnv, make_env
from aux_tasks.data.replay_buffer import ReplayBuffer
from aux_tasks.utils.logging import WandBLogger


def fix_config(cfg: DictConfig, env: Env):
    action_shape = env.get_action_space()[-1]
    obs_shape = env.get_observation_space()[-1]
    cfg.models.actor.out_dim = action_shape
    try:
        cfg.models.decoder.out_dim = obs_shape
    except ConfigAttributeError:
        pass
    try:
        cfg.models.critic.out_dim = action_shape
    except ConfigAttributeError:
        pass

    # handles logging correctly
    if not cfg.alt_path:
        cfg.alt_path = os.getcwd()
    return cfg


@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    # debug options for jitting
    config.update("jax_disable_jit", cfg.debug)

    # building config objects
    train_cfg: TrainHyperparams = OmegaConf.to_object(cfg.train)  # type: ignore
    env_cfg: EnvConfig = OmegaConf.to_object(cfg.env)  # type: ignore
    algo_cfg: AlgoHyperparams = OmegaConf.to_object(cfg.algo)  # type: ignore

    env = make_env(env_cfg)

    fixed_cfg = fix_config(cfg, env)

    train_cfg: TrainHyperparams = OmegaConf.to_object(fixed_cfg.train)  # type: ignore
    env_cfg: EnvConfig = OmegaConf.to_object(fixed_cfg.env)  # type: ignore
    algo_cfg: AlgoHyperparams = OmegaConf.to_object(fixed_cfg.algo)  # type: ignore
    cfg = fix_config(cfg, env)

    # setup models
    critic = hydra.utils.instantiate(fixed_cfg.models.critic)
    critic_target = hydra.utils.instantiate(fixed_cfg.models.critic)
    actor = hydra.utils.instantiate(fixed_cfg.models.actor)
    encoder = hydra.utils.instantiate(fixed_cfg.models.encoder)
    encoder_target = hydra.utils.instantiate(fixed_cfg.models.encoder)
    latent_model = hydra.utils.instantiate(fixed_cfg.models.latent_model)
    decoder = hydra.utils.instantiate(fixed_cfg.models.decoder)

    replay_buffer = ReplayBuffer(
        env.get_observation_space(),
        env.get_action_space(),
        train_cfg.num_seeds,
        train_cfg.total_steps,
        algo_cfg.length_training_rollout,
        is_img_obs=isinstance(env, MinAtarEnv),
    )

    logger = WandBLogger(
        cfg.logger.project,
        cfg.logger.entity,
        OmegaConf.to_container(cfg, resolve=True),
        wandb_init_path=os.path.join(cfg.alt_path, "wandb_init.txt"),
    )
    if cfg.debug:
        logger = WandBLogger(
            "debug_{}".format(cfg.logger.project),
            cfg.logger.entity,
            OmegaConf.to_container(cfg, resolve=True),
            debug=cfg.debug,
        )

    if train_cfg.num_seeds > 1:
        trainer = MultiSeedTrainer(
            critic,
            critic_target,
            actor,
            encoder,
            encoder_target,
            latent_model,
            decoder,
            replay_buffer,
            env,
            algo_cfg,
            train_cfg,
            logger,
        )
    else:
        trainer = SingleSeedTrainer(
            critic,
            critic_target,
            actor,
            encoder,
            encoder_target,
            latent_model,
            decoder,
            replay_buffer,
            env,
            algo_cfg,
            train_cfg,
            logger,
        )

    trainer.check_pretrain("checkpoint", cfg.alt_path)
    trainer.train()


if __name__ == "__main__":
    main()
