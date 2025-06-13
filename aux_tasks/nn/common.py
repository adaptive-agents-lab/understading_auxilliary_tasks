from typing import Callable, Sequence, Optional
import math

import flax.linen as nn
import jax
from jax.nn import initializers

from aux_tasks.utils.jax import torch_he_uniform


class NormLayer(nn.Module):
    size: int
    activation: Optional[Callable[[jax.Array], jax.Array]]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(
            self.size,
            # kernel_init=torch_he_uniform(),
            # bias_init=initializers.uniform(1 / math.sqrt(self.size)),
        )(x)
        x = nn.LayerNorm()(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ActivationLayer(nn.Module):
    size: int
    activation: Optional[Callable[[jax.Array], jax.Array]]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(
            self.size,
            # kernel_init=torch_he_uniform(),
            # bias_init=initializers.uniform(1 / math.sqrt(self.size)),
        )(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    activation_input: Optional[Callable[[jax.Array], jax.Array]]
    activation_output: Optional[Callable[[jax.Array], jax.Array]]
    activations_hidden: Optional[Callable[[jax.Array], jax.Array]]
    normalize_input: bool
    normalize_output: bool
    normalize_hidden: bool

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        inp_layer_size = self.hidden_dims[0]
        hidden_layer_sizes = self.hidden_dims[1:]
        output_layer_size = self.out_dim
        if self.normalize_input:
            x = NormLayer(inp_layer_size, self.activation_input)(x)
        else:
            x = ActivationLayer(inp_layer_size, self.activation_input)(x)
        for size in hidden_layer_sizes:
            if self.normalize_hidden:
                x = NormLayer(size, self.activations_hidden)(x)
            else:
                x = ActivationLayer(size, self.activations_hidden)(x)
        if self.normalize_output:
            x = NormLayer(output_layer_size, self.activation_output)(x)
        else:
            x = ActivationLayer(output_layer_size, self.activation_output)(x)
        return x
