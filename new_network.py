# %%
from jaxrl.networks.policies import MetaPolicy

class NewMetaPolicy(MetaPolicy):
    def setup(self):
        super().setup()

    def __call__(self, x, t, temperature=1.0):
        x = super().__call__(x, t, temperature)
        return x
# %%
# %%
import os 
os.environ["MUJOCO_GL"] = "egl"
from continual_world import TASK_SEQS, get_single_env
import numpy as np
import jax.numpy as jnp
import random
import matplotlib.pyplot as plt
from jaxrl.networks import policies
# %%
env_name = "cw1-hammer"
seed = 110
randomization = "random_init_all"
normalize_reward = True
seq_tasks = TASK_SEQS[env_name]
env = get_single_env(name=seq_tasks[0]['task'], seed=seed, randomization=randomization, normalize_reward=normalize_reward)
# %%
actor_configs = {
    "clip_mean": 1.0,
    "final_fc_init_scale": 1.0e-4,
    "hidden_dims": [1024, 1024, 1024, 1024],
    "name_activation": "leaky_relu",
    "state_dependent_std": True,
    "use_layer_norm": True,
    "action_dim": 4,
    "task_num": 10,
}
# %%
pi_opt_configs = {
  "optim_algo": "adam",
  "clip_method": None,
  "max_norm": -1,
  "opt_kargs": {
    "learning_rate": 3.0e-4}
}
# %%
import jax
from flax.core import FrozenDict
from jaxrl.agents.sac.sac_learner import MPNTrainState
import jax.numpy as jnp
seed = 0
actor_def = NewMetaPolicy(**actor_configs)
actor_params = FrozenDict(actor_def.init(jax.random.PRNGKey(seed+1), jnp.zeros((1, env.observation_space.shape[0])), jnp.array([0])).pop('params'))
actor = MPNTrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=utils_fn.set_optimizer(**pi_opt_configs))
# %%