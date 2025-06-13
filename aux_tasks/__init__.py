from hydra.core.config_store import ConfigStore
from aux_tasks.agents.agent_config import AlgoHyperparams

from aux_tasks.trainers import TrainHyperparams
from aux_tasks.data.env import EnvConfig

cs = ConfigStore.instance()
cs.store(group="train", name="base_train", node=TrainHyperparams)
cs.store(group="env", name="base_env", node=EnvConfig)
cs.store(group="algo", name="base_algo", node=AlgoHyperparams)
