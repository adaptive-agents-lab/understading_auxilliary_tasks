from jax import random

from flax import linen as nn


layer = nn.Conv(64, (3, 3), padding="VALID")

k = random.PRNGKey(0)

x = random.normal(k, (40, 3, 3, 3))

params = layer.init(k, x)

y = layer.apply(params, x)

print(y.shape)
