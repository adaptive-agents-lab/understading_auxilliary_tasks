from aux_tasks.data.env import GymnaxEnvConfig
from aux_tasks.data.env import GymnaxEnv

import jax

rng = jax.random.PRNGKey(0)
rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

conf = GymnaxEnvConfig(name="Asterix-MinAtar")

env = GymnaxEnv(conf)

env_reset = env.get_reset()
env_step = env.get_step()

state = env_reset(key_reset)

obs_space = env.get_observation_space()
print(obs_space)
act_space = env.get_action_space()
print(act_space)

print(env.sample_action_space(key_reset))
