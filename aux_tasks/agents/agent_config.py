from dataclasses import dataclass
from typing import Callable, Dict, List

from flax.training.train_state import TrainState
from flax import struct


@dataclass
class LossFunctions:
    loss_functions: Dict[str, Callable]

    def __hash__(self):
        return hash(self.loss_functions.keys())

    def __eq__(self, __value: object) -> bool:
        return hash(self) == hash(__value)

    def __repr__(self) -> str:
        return str(self.loss_functions.keys())


@dataclass
class AlgoHyperparams:
    actor_learning_rate: float
    critic_learning_rate: float
    encoder_learning_rate: float
    model_learning_rate: float
    gradient_clip: float
    gamma: float
    tau: float
    td_average: bool
    length_mve: int
    length_training_rollout: int
    muzero_direct_vf_target: bool
    vaml_real_reward: bool
    use_muzero_critic_update: bool
    add_model_noise: bool
    use_mve: bool
    use_svg: bool
    detach_actor_encoder: bool
    detach_critic_encoder: bool
    loss_functions: List[str]

    def __hash__(self):
        return hash(
            (
                self.actor_learning_rate,
                self.critic_learning_rate,
                self.encoder_learning_rate,
                self.model_learning_rate,
                self.gradient_clip,
                self.gamma,
                self.td_average,
                self.length_mve,
                self.length_training_rollout,
                self.vaml_real_reward,
                self.muzero_direct_vf_target,
                self.use_muzero_critic_update,
                self.use_mve,
                self.use_svg,
                self.detach_actor_encoder,
                self.detach_critic_encoder,
            )
        )

    def replace(self, **kwargs):
        return AlgoHyperparams(**{**self.__dict__, **kwargs})


@struct.dataclass
class Models:
    critic: TrainState
    critic_target: TrainState
    actor: TrainState
    encoder: TrainState
    encoder_target: TrainState
    decoder: TrainState
    latent_model: TrainState
