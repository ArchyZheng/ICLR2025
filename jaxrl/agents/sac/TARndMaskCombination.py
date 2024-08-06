from jax.numpy import ndarray
from numpy import ndarray
from jaxrl.agents.sac.sac_learner import CoTASPLearner
from jaxrl.networks.policies import ste_step_fn
from jaxrl.agents.sac.sac_mask_combination import MaskCombinationLearner
import jax.numpy as jnp
from jaxrl.datasets.dataset import Batch
from jaxrl.networks.policies_PRE import Decoder_PRE
from jaxrl.networks.rnd_net import rnd_network
import jax
from flax.core import FrozenDict, unfreeze, freeze
from jaxrl.networks.common import TrainState, PRNGKey, Params, InfoDict, MPNTrainState
from optax import global_norm
import jaxrl.networks.common as utils_fn
from typing import Any, Tuple
import functools
from jaxrl.agents.sac.sac_learner import _update_critic, _update_temp
import numpy as np
from jaxrl.dict_learning.task_dict import OnlineDictLearnerV2

def _get_intrinsic_reward(rnd_net: rnd_network, rnd_parameter: FrozenDict, obs: jnp.ndarray, act: jnp.ndarray, next_obs: jnp.ndarray,task_id: jnp.ndarray) -> jnp.ndarray:
    target_next_st = rnd_net.apply(rnd_parameter, next_obs, task_id)
    return target_next_st

class TARndMaskCombinationLearner(MaskCombinationLearner):
    def __init__(self, seed: int, observations: jnp.ndarray, actions: jnp.ndarray, task_num: int, load_policy_dir: str | None = None, load_dict_dir: str | None = None, update_dict=True, update_coef=True, dict_configs_fixed: dict = ..., dict_configs_random: dict = ..., pi_opt_configs: dict = ..., q_opt_configs: dict = ..., t_opt_configs: dict = ..., actor_configs: dict = ..., critic_configs: dict = ..., tau: float = 0.005, discount: float = 0.99, target_update_period: int = 1, target_entropy: float | None = None, init_temperature: float = 1):
        super().__init__(seed, observations, actions, task_num, load_policy_dir, load_dict_dir, update_dict, update_coef, dict_configs_fixed, dict_configs_random, pi_opt_configs, q_opt_configs, t_opt_configs, actor_configs, critic_configs, tau, discount, target_update_period, target_entropy, init_temperature)
        decoder_key, rnd_key = jax.random.split(jax.random.PRNGKey(seed + 599), 2)

        decoder_def = Decoder_PRE()
        decoder_params = FrozenDict(decoder_def.init(decoder_key, jnp.ones((1, 1024 + ...))).pop('params')) # I should verify this.
        # I should verify this.
        decoder_network = TrainState.create(
            apply_fn=decoder_def.apply,
            params=decoder_params,
            tx=utils_fn.set_optimizer(**t_opt_configs)
        )

        self.t_opt_configs = t_opt_configs
        self.decoder = decoder_network

        self.rnd_net = rnd_network()
        self.rnd_net_params = FrozenDict(self.rnd_net.init(rnd_key, jnp.ones((1, 12)), jnp.ones((1, 4, 1024, 1))).pop('params'))
    
    def start_task(self, task_id: int, description: str):
        super().start_task(task_id, description)

        actor_params = self.actor.params
        task_embedding = []
        for k in actor_params.keys():
            if k.startswith('embeds'):
                task_embedding.append(actor_params[k]['embedding'])
        task_embedding_alpha = jnp.stack(task_embedding, axis=0)
        task_embedding_mask = ste_step_fn(task_embedding_alpha)
        task_embedding_mask = jnp.expand_dims(task_embedding_mask, axis=[0, 3])
        self.rnd_net.task_embedding_mask = task_embedding_mask

        



        


        
        
        
        