from jax.numpy import ndarray
import jax.numpy as jnp
from jaxrl.agents.sac.sac_learner import CoTASPLearner
# from sac_learner import CoTASPLearner
from typing import Any
import numpy as np
from jaxrl.dict_learning.task_dict import OnlineDictLearnerV2
from flax.core import freeze, unfreeze, FrozenDict
import jax

class RandomGenerateD(OnlineDictLearnerV2):
    pass


class MaskCombinationLearner(CoTASPLearner):
    def __init__(self, seed: int, observations: jnp.ndarray, actions: jnp.ndarray, task_num: int, load_policy_dir: str | None = None, load_dict_dir: str | None = None, update_dict=True, update_coef=True, dict_configs_fixed: dict = ..., dict_configs_random: dict = ..., pi_opt_configs: dict = ..., q_opt_configs: dict = ..., t_opt_configs: dict = ..., actor_configs: dict = ..., critic_configs: dict = ..., tau: float = 0.005, discount: float = 0.99, target_update_period: int = 1, target_entropy: float | None = None, init_temperature: float = 1):
        super().__init__(seed, observations, actions, task_num, load_policy_dir, load_dict_dir, update_dict, update_coef, dict_configs_fixed, pi_opt_configs, q_opt_configs, t_opt_configs, actor_configs, critic_configs, tau, discount, target_update_period, target_entropy, init_temperature)
        # fixed dictionary this has been integrated by CoTASP without any dictionary learning
        # random dictionary
        self.dict4layers_random = {}
        self.actor_configs = actor_configs
        self.dict_configs_random = dict_configs_random
        
        
    
    def start_task(self, task_id: int, description: str):
        task_e = self.task_encoder.encode(description)[np.newaxis, :]
        self.task_embeddings.append(task_e)

        # set initial alpha for each layer of MPN
        actor_params = unfreeze(self.actor.params)
        for k in self.actor.params.keys():
            if k.startswith('embeds'):
                alpha_l = self.dict4layers[k].get_alpha(task_e)
                alpha_l = jnp.asarray(alpha_l.flatten())
                # Replace the i-th row
                actor_params[k]['embedding'] = actor_params[k]['embedding'].at[task_id].set(alpha_l)

        
        actor_configs = self.actor_configs
        dict_configs = self.dict_configs_random
        for id_layer, hidn in enumerate(actor_configs['hidden_dims']):
            dict_learner = OnlineDictLearnerV2(
                384,
                hidn,
                task_id * 10000 + id_layer + 10000,
                None,
                **dict_configs)
            self.dict4layers_random[f'random_embeds_bb_{id_layer}'] = dict_learner
        
        for k in self.actor.params.keys():
            if k.startswith('random'):
                alpha_l = self.dict4layers_random[k].get_alpha(task_e)
                alpha_l = jnp.asarray(alpha_l.flatten())
                # Replace the i-th row
                actor_params[k]['embedding'] = actor_params[k]['embedding'].at[task_id].set(alpha_l)
        self.actor = self.actor.update_params(freeze(actor_params))
        # self.rng, _, dicts = _sample_actions(
        #     self.rng, self.actor, self.dummy_o, jnp.array([task_id])
        # )
        # self.task_mask = get_task_mask(dicts['masks'])
        # # store task_mask as npy file
        # with open(f'./mask_npy/task_mask_{task_id}.npy', 'wb') as f:
        #     np.save(f, self.task_mask)

def get_task_mask(masks):
    current_mask = {}
    for key, value in masks.items():
        current_mask[key] = value[0]
    candidate_list = ['backbones_0', 'backbones_1', 'backbones_2', 'backbones_3']
    output_list = []
    for name in candidate_list:
        output_list.append(current_mask[name])
    output_list = jnp.stack(output_list)
    output_list = jnp.expand_dims(output_list, axis=[0, 3])
    return output_list

@jax.jit
def _sample_actions(
    rng,
    actor,
    observations: np.ndarray,
    task_i: jnp.ndarray,
    temperature: float = 1.0
    ):
    
    rng, key = jax.random.split(rng)
    dist, dicts = actor(
        observations,
        task_i,
        temperature
    )
    return rng, dist.sample(seed=key), dicts

    
    
    
