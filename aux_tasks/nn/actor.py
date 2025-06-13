from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from aux_tasks.nn.common import MLP
from aux_tasks.utils.jax import mish


class TanhActor(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    activations_hidden: Callable[[jax.Array], jax.Array]
    activation_input: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_hidden: bool

    @nn.compact
    def __call__(self, states):
        x = MLP(
            out_dim=self.out_dim,
            hidden_dims=self.hidden_dims,
            activations_hidden=self.activations_hidden,
            activation_input=self.activation_input,
            activation_output=jnp.tanh,
            normalize_input=self.normalize_input,
            normalize_output=False,
            normalize_hidden=self.normalize_hidden,
        )(states)
        return x


class TDMPCActor(TanhActor):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Callable[[jax.Array], jax.Array] = jax.nn.elu
    normalize_input: bool = False
    normalize_hidden: bool = False


class TDMPC2Actor(TanhActor):
    activations_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Callable[[jax.Array], jax.Array] = mish
    normalize_input: bool = True
    normalize_hidden: bool = True


class ViperActor(TanhActor):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Callable[[jax.Array], jax.Array] = jax.nn.elu
    normalize_input: bool = False
    normalize_hidden: bool = False
