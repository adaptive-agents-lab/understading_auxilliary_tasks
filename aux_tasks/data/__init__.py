from hydra.core.config_store import ConfigStore

from aux_tasks.data.env import (
    EnvConfig,
    GymnaxEnvConfig,
    BraxEnvConfig,
    DMCEnvConfig,
    MinAtarEnvConfig,
    MultiRewardDMCEnvConfig,
)

cs = ConfigStore.instance()
cs.store(group="env", name="base_env", node=EnvConfig)
cs.store(group="env", name="gymnax_env", node=GymnaxEnvConfig)
cs.store(group="env", name="minatar_env", node=MinAtarEnvConfig)
cs.store(group="env", name="brax_env", node=BraxEnvConfig)
cs.store(group="env", name="dmc_env", node=DMCEnvConfig)
cs.store(group="env", name="multi_rew_dmc_env", node=MultiRewardDMCEnvConfig)
