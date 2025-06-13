import abc
from typing import Tuple, Union

import jax
import flax.linen as nn
from flax import struct


class AbstractCritic(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, states: jax.Array, actions: jax.Array
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        pass


class AbstractActor(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, states: jax.Array, actions: jax.Array) -> jax.Array:
        pass


class AbstractEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, states: jax.Array) -> jax.Array:
        pass


class AbstractLatentModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, states: jax.Array, actions: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        pass


class AbstractDecoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, states: jax.Array) -> jax.Array:
        pass


class AbstractBatch(abc.ABC):
    pass


@struct.dataclass
class RLBatch(AbstractBatch):
    state: jax.Array
    action: jax.Array
    reward: jax.Array
    next_state: jax.Array
    mask: jax.Array


class Dataset(abc.ABC):
    @abc.abstractmethod
    def sample(self, batch_size: int, key: int) -> AbstractBatch:
        pass

    @abc.abstractmethod
    def load(self, path: str):
        pass

    @abc.abstractmethod
    def save(self, path: str):
        pass
