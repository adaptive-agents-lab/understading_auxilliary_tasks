from typing import Callable, Optional, Sequence, Dict, Union
import flax
from flax.training import train_state
import jax
import jax.numpy as jnp


PyTree = Union[jax.Array, Dict[str, "PyTree"]]


def nn_flatten(x: jax.Array, axis: int) -> jax.Array:
    return jax.lax.collapse(x, axis)


def batch_loss_fn(
    loss_fn: Callable,
    in_axes: Sequence[Union[int, None]] = (),
    out_axes: Sequence[Union[int, None]] = (),
    has_aux: bool = False,
) -> Callable:
    _batched_loss_fn = jax.vmap(loss_fn, in_axes=in_axes, out_axes=out_axes)

    def _f(*args):
        if has_aux:
            value, aux = _batched_loss_fn(*args)
            return jnp.mean(value), jax.tree_map(lambda x: jnp.mean(x), aux)
        else:
            value = _batched_loss_fn(*args)
            return jnp.mean(value)

    return _f


# @jax.jit
def tree_list_mean(lot: Sequence[PyTree]) -> PyTree:
    return jax.tree_map(lambda *x: jnp.mean(jnp.stack(x, axis=0), axis=0), *lot)


def torch_he_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype=jnp.float_,
):
    "TODO: push to jax"
    return jax.nn.initializers.variance_scaling(
        0.3333,
        "fan_in",
        "uniform",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def clamp_int(i, min_value, max_value):
    """Clamp an integer between min_value and max_value.

    Args:
        i (int): The integer to clamp.
        min_value (int): The minimum value.
        max_value (int): The maximum value.

    Returns:
        int: The clamped integer.
    """
    return int(max(min(i, max_value), min_value))


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class ExpandedTrainState(train_state.TrainState):
    variables: Optional[PyTree] = flax.core.frozen_dict.FrozenDict({})


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def multi_softmax(x, dim=8):
    return jax.nn.softmax(x.reshape(-1, dim), axis=-1).reshape(x.shape)
