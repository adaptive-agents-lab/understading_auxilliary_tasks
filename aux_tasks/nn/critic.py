from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from aux_tasks.nn.common import MLP
from aux_tasks.rl_types import AbstractCritic
from aux_tasks.utils.jax import mish


class Critic(AbstractCritic):
    hidden_dims: Sequence[int]
    activation_input: Callable[[jax.Array], jax.Array]
    activations_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_hidden: bool
    out_dim: int

    @nn.compact
    def __call__(self, states, actions):
        x = jnp.concatenate([states, actions], axis=-1)
        x = MLP(
            out_dim=self.out_dim,
            hidden_dims=self.hidden_dims,
            activations_hidden=self.activations_hidden,
            activation_input=self.activation_input,
            activation_output=None,
            normalize_input=self.normalize_input,
            normalize_output=False,
            normalize_hidden=self.normalize_hidden,
        )(x)
        return x


class TwinnedCritic(AbstractCritic):
    hidden_dims: Sequence[int]
    activation_input: Callable[[jax.Array], jax.Array]
    activations_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_hidden: bool

    @nn.compact
    def __call__(self, states, actions):
        x1 = Critic(
            hidden_dims=self.hidden_dims,
            activations_hidden=self.activations_hidden,
            activation_input=self.activation_input,
            normalize_input=self.normalize_input,
            normalize_hidden=self.normalize_hidden,
            out_dim=1,
        )(states, actions)
        x2 = Critic(
            hidden_dims=self.hidden_dims,
            activations_hidden=self.activations_hidden,
            activation_input=self.activation_input,
            normalize_input=self.normalize_input,
            normalize_hidden=self.normalize_hidden,
            out_dim=1,
        )(states, actions)
        return x1, x2


class DuellingDoubleCritic(AbstractCritic):
    hidden_dims: Sequence[int]
    activation_input: Callable[[jax.Array], jax.Array]
    activations_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_hidden: bool
    out_dim: int

    @nn.compact
    def __call__(self, states: jax.Array, actions: jax.Array | None = None):
        v = MLP(
            hidden_dims=self.hidden_dims,
            activations_hidden=self.activations_hidden,
            activation_input=self.activation_input,
            normalize_input=self.normalize_input,
            normalize_hidden=self.normalize_hidden,
            out_dim=1,
            activation_output=None,
            normalize_output=False,
        )(states)
        a = MLP(
            hidden_dims=self.hidden_dims,
            activations_hidden=self.activations_hidden,
            activation_input=self.activation_input,
            normalize_input=self.normalize_input,
            normalize_hidden=self.normalize_hidden,
            out_dim=self.out_dim,
            activation_output=None,
            normalize_output=False,
        )(states)
        return v + (a - jnp.mean(a, axis=-1, keepdims=True))


class DQNCritic(AbstractCritic):
    hidden_dims: Sequence[int]
    activation_input: Callable[[jax.Array], jax.Array]
    activations_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_hidden: bool
    out_dim: int

    @nn.compact
    def __call__(self, states: jax.Array, actions: jax.Array | None = None):
        return MLP(
            hidden_dims=self.hidden_dims,
            activations_hidden=self.activations_hidden,
            activation_input=self.activation_input,
            normalize_input=self.normalize_input,
            normalize_hidden=self.normalize_hidden,
            out_dim=self.out_dim,
            activation_output=None,
            normalize_output=False,
        )(states)


class TDMPCCritic(TwinnedCritic):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Callable[[jax.Array], jax.Array] = jax.nn.tanh
    normalize_input: bool = True
    normalize_hidden: bool = False


class TDMPC2Critic(TwinnedCritic):
    activations_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Callable[[jax.Array], jax.Array] = mish
    normalize_input: bool = True
    normalize_hidden: bool = True


class ViperCritic(TwinnedCritic):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.tanh
    activation_input: Callable[[jax.Array], jax.Array] = jax.nn.elu
    normalize_input: bool = False
    normalize_hidden: bool = True


class ViperDuellingDoubleCritic(DuellingDoubleCritic):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.tanh
    activation_input: Callable[[jax.Array], jax.Array] = jax.nn.elu
    normalize_input: bool = False
    normalize_hidden: bool = True
    out_dim: int = 1


class ViperDQNCritic(DuellingDoubleCritic):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.tanh
    activation_input: Callable[[jax.Array], jax.Array] = jax.nn.elu
    normalize_input: bool = False
    normalize_hidden: bool = True
    out_dim: int = 1
