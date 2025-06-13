from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from aux_tasks.nn.common import MLP, ActivationLayer
from aux_tasks.utils.jax import mish, multi_softmax, nn_flatten


def LayerNormTanh(x):
    return nn.LayerNorm(use_bias=False, use_scale=False)(x)


class MLPEncoder(nn.Module):
    hidden_dims: Sequence[int]
    feature_dim: int
    activation_input: Optional[Callable[[jax.Array], jax.Array]]
    activation_output: Optional[Callable[[jax.Array], jax.Array]]
    activations_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_output: bool
    normalize_hidden: bool

    @nn.compact
    def __call__(self, states):
        return MLP(
            hidden_dims=self.hidden_dims,
            out_dim=self.feature_dim,
            activations_hidden=self.activations_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
        )(states)


class ConvEncoder(nn.Module):
    conv_kernel: Sequence[int]
    conv_filters: int
    hidden_dims: Sequence[int]
    feature_dim: int
    activation_input: Callable[[jax.Array], jax.Array] | None
    activation_output: Callable[[jax.Array], jax.Array] | None
    activations_hidden: Callable[[jax.Array], jax.Array] | None
    normalize_input: bool
    normalize_output: bool
    normalize_hidden: bool
    conv_stride: Sequence[int] | int = 1

    @nn.compact
    def __call__(self, states):
        x = nn.Conv(
            features=self.conv_filters,
            kernel_size=self.conv_kernel,
            strides=self.conv_stride,
            padding="VALID",
        )(states)
        x = self.activations_hidden(x)
        # x = nn.Conv(
        #     features=self.conv_filters,
        #     kernel_size=self.conv_kernel,
        #     strides=self.conv_stride,
        #     padding='VALID',
        # )(x)
        # x = self.activations_hidden(x)
        x = nn_flatten(x, axis=-3)
        x = ActivationLayer(
            self.feature_dim,
            activation=self.activation_output,
        )(x)
        return x


class ConvDecoder(nn.Module):
    out_dim: int
    conv_kernel: Sequence[int]
    conv_stride: Sequence[int]
    conv_filters: int
    hidden_dims: Sequence[int]
    activation_input: Optional[Callable[[jax.Array], jax.Array]]
    activation_output: Optional[Callable[[jax.Array], jax.Array]]
    activations_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_output: bool
    normalize_hidden: bool

    @nn.compact
    def __call__(self, states):
        x = ActivationLayer(
            self.conv_filters * 10 * 10, activation=self.activation_input
        )(states)
        x = x.reshape((-1, 10, 10, self.conv_filters))
        # x = nn.ConvTranspose(
        #     features=self.conv_filters,
        #     kernel_size=self.conv_kernel,
        #     strides=self.conv_stride,
        #     padding='SAME',
        # )(x)
        # x = self.activations_hidden(x)
        x = nn.ConvTranspose(
            features=self.out_dim,
            kernel_size=self.conv_kernel,
            strides=self.conv_stride,
            padding="SAME",
        )(x)
        return x


class MLPDecoder(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    activations_hidden: Callable[[jax.Array], jax.Array]
    activation_input: Optional[Callable[[jax.Array], jax.Array]]
    activation_output: Optional[Callable[[jax.Array], jax.Array]]
    normalize_input: bool
    normalize_output: bool
    normalize_hidden: bool

    @nn.compact
    def __call__(self, states):
        return MLP(
            hidden_dims=self.hidden_dims,
            out_dim=self.out_dim,
            activations_hidden=self.activations_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
        )(states)


class LatentModel(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    activations: Callable[[jax.Array], jax.Array] = mish
    activation_final: Callable[[jax.Array], jax.Array] = multi_softmax
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = None
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    activations_hidden: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = False
    normalize_output: bool = False
    normalize_hidden: bool = False

    @nn.compact
    def __call__(self, states, actions):
        reward = MLP(
            out_dim=1,
            hidden_dims=self.hidden_dims,
            activations_hidden=self.activations_hidden,
            activation_input=self.activation_input,
            activation_output=None,
            normalize_input=self.normalize_input,
            normalize_output=False,
            normalize_hidden=self.normalize_hidden,
        )
        forward = MLP(
            hidden_dims=self.hidden_dims,
            out_dim=self.out_dim,
            activations_hidden=self.activations_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
        )

        x = jnp.concatenate([states, actions], axis=-1)
        forward = forward(x)
        rew = reward(x)
        return forward, rew


class TDMPCEncoder(MLPEncoder):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = jax.nn.elu
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = False
    normalize_output: bool = False
    normalize_hidden: bool = False


class ViperEncoder(MLPEncoder):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = jax.nn.elu
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = jax.nn.elu
    normalize_input: bool = False
    normalize_output: bool = False
    normalize_hidden: bool = False


class ViperConvEncoder(ConvEncoder):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = jax.nn.elu
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = jax.nn.elu
    normalize_input: bool = False
    normalize_output: bool = False
    normalize_hidden: bool = False
    conv_kernel: Sequence[int] = (3, 3)
    conv_filters: int = 16
    hidden_dims: Sequence[int] = ()
    feature_dim: int = 128


class ViperDecoder(MLPDecoder):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = jax.nn.elu
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = False
    normalize_output: bool = False
    normalize_hidden: bool = False


class ViperConvDecoder(ConvDecoder):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = jax.nn.elu
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = False
    normalize_output: bool = False
    normalize_hidden: bool = False
    conv_kernel: Sequence[int] = (3, 3)
    conv_stride: Sequence[int] = (1, 1)
    conv_filters: int = 16
    hidden_dims: Sequence[int] = ()
    feature_dim: int = 128


class TDMPC2Encoder(MLPEncoder):
    activations_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = multi_softmax
    normalize_input: bool = True
    normalize_output: bool = True
    normalize_hidden: bool = True


class TDMPC2ContinuousEncoder(MLPEncoder):
    activations_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = mish
    normalize_input: bool = True
    normalize_output: bool = False
    normalize_hidden: bool = True


class TDMPCLatentModel(LatentModel):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = jax.nn.elu
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = False
    normalize_output: bool = False
    normalize_hidden: bool = False


class ViperLatentModel(LatentModel):
    activations_hidden: Callable[[jax.Array], jax.Array] = jax.nn.elu
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = jax.nn.elu
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = False
    normalize_output: bool = False
    normalize_hidden: bool = False


class TDMPC2LatentModel(LatentModel):
    activations_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = multi_softmax
    normalize_input: bool = True
    normalize_output: bool = True
    normalize_hidden: bool = True


class TDMPC2ContinuousLatentModel(LatentModel):
    activations_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = True
    normalize_output: bool = False
    normalize_hidden: bool = True
