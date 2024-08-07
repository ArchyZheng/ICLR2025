from jax.numpy import ndarray
from numpy import ndarray
from jaxrl.agents.sac.sac_learner import CoTASPLearner
from jaxrl.networks.policies import ste_step_fn
from jaxrl.agents.sac.sac_mask_combination import MaskCombinationLearner
import jax.numpy as jnp
from jaxrl.datasets.dataset import Batch
from jaxrl.networks.policies_PRE import Decoder_PRE
from jaxrl.networks.rnd_net import rnd_network
from jaxrl.agents.sac.critic import target_update
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


@functools.partial(jax.jit, static_argnames=('update_target'))
def _update_cotasp_jit(rng: PRNGKey, task_id: int, tau: float, discount: float, 
    target_entropy: float, optimize_alpha: bool, param_mask: FrozenDict[str, Any], 
    actor: MPNTrainState, critic: TrainState, target_critic: TrainState, 
    update_target, temp: TrainState, batch: Batch, decoder: TrainState, rnd_net: TrainState, ext_coeff: float, int_coeff: float, decoder_rate: float = 1.0, rnd_rate: float = 1.0
    ) -> Tuple[PRNGKey, MPNTrainState, TrainState, TrainState, TrainState, InfoDict]:
    # optimizing critics 
    new_rng, new_critic, new_target_critic, critic_info = _update_critic(
        rng, task_id, actor, critic, target_critic, update_target, temp, 
        batch, discount, tau, decoder, rnd_net, ext_coeff, int_coeff
    )

    # optimizing either alpha or theta
    new_rng, new_actor, actor_info, new_decoder = _update_theta(new_rng, task_id, param_mask, actor, new_critic, temp, batch, decoder, rnd_net, rnd_rate)

    # updating temperature coefficient
    new_temp, temp_info = _update_temp(
        temp, actor_info['entropy'], target_entropy
    )

    return new_rng, new_actor, new_temp, new_critic, new_target_critic, {
        **actor_info,
        **temp_info,
        **critic_info
    }, new_decoder

def _update_theta(
    rng: PRNGKey, task_id: int, param_mask: FrozenDict[str, Any], 
    actor: MPNTrainState, critic: TrainState, temp: TrainState, 
    batch: Batch, decoder: TrainState, rnd_net: TrainState, rnd_rate) -> Tuple[PRNGKey, MPNTrainState, InfoDict]:

    rng, key = jax.random.split(rng)
    def actor_decoder_loss_fn(actor_params: Params, decoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, dicts = actor.apply_fn(
            {'params': actor_params}, batch.observations, jnp.array([task_id])
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()

        # >>>>>>>>>>>>>>>> update decoder >>>>>>>>>>>>>>>>>>>>>>>
        encoder_output = dicts['encoder_output']
        encoder_output = jax.lax.stop_gradient(encoder_output)
        decoder_input = jnp.concatenate([encoder_output, batch.actions], -1)
        predict_embedding = decoder.apply_fn({'params': decoder_params}, decoder_input)
        target_embedding = rnd_net.apply_fn({'params': rnd_net.params}, batch.observations)
        rnd_loss = jnp.mean(jnp.square(target_embedding - predict_embedding))
        # <<<<<<<<<<<<<<<< update decoder <<<<<<<<<<<<<<<<<<<<<<<

        _info = {
            'rnd_loss': rnd_loss,
            'hac_sac_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'means': dicts['means'].mean()
        }
        for k in dicts['masks']:
            _info[k+'_rate_act'] = jnp.mean(dicts['masks'][k])

        return actor_loss + rnd_rate * rnd_loss, _info
    grads_actor_decoder, actor_info = jax.grad(actor_decoder_loss_fn, has_aux=True, argnums=[0, 1])(actor.params, decoder.params)
    grads_actor, grads_decoder = grads_actor_decoder
    # recording info
    g_norm = global_norm(grads_actor)
    actor_info['g_norm_actor'] = g_norm
    for p, v in param_mask.items():
        if p[-1] == 'kernel':
            actor_info['used_capacity_'+p[0]] = 1.0 - jnp.mean(v)

    # Masking gradients according to cumulative binary masks
    unfrozen_grads = unfreeze(grads_actor)
    for path, value in param_mask.items():
        cursor = unfrozen_grads
        for key in path[:-1]:
            if key in cursor:
                cursor = cursor[key]
        cursor[path[-1]] *= value
    
    # only update policy parameters (theta)
    new_actor = actor.apply_grads_theta(grads=freeze(unfrozen_grads))
    new_decoder = decoder.apply_gradients(grads=grads_decoder)

    return rng, new_actor, actor_info, new_decoder 


@jax.jit
def _get_intrinsic_reward(rnd_net: rnd_network, rnd_parameter: FrozenDict, next_obs: jnp.ndarray,task_id: jnp.ndarray) -> jnp.ndarray:
    target_next_st = rnd_net.apply(rnd_parameter, next_obs, task_id)
    return target_next_st

class TARndMaskCombinationLearner(MaskCombinationLearner):
    def __init__(self, seed: int, observations: jnp.ndarray, actions: jnp.ndarray, task_num: int, load_policy_dir: str | None = None, load_dict_dir: str | None = None, update_dict=True, update_coef=True, dict_configs_fixed: dict = ..., dict_configs_random: dict = ..., pi_opt_configs: dict = ..., q_opt_configs: dict = ..., t_opt_configs: dict = ..., actor_configs: dict = ..., critic_configs: dict = ..., tau: float = 0.005, discount: float = 0.99, target_update_period: int = 1, target_entropy: float | None = None, init_temperature: float = 1, ext_coeff: float = 1.0, int_coeff: float = 1.0, rnd_rate: float = 1.0):
        super().__init__(seed, observations, actions, task_num, load_policy_dir, load_dict_dir, update_dict, update_coef, dict_configs_fixed, dict_configs_random, pi_opt_configs, q_opt_configs, t_opt_configs, actor_configs, critic_configs, tau, discount, target_update_period, target_entropy, init_temperature)
        decoder_key, rnd_key = jax.random.split(jax.random.PRNGKey(seed + 599), 2)

        decoder_def = Decoder_PRE()
        decoder_params = FrozenDict(decoder_def.init(decoder_key, jnp.ones((1, 1024 + 4))).pop('params')) # 1024: state dim, 4: action dim
        # I should verify this.
        decoder_network = TrainState.create(
            apply_fn=decoder_def.apply,
            params=decoder_params,
            tx=utils_fn.set_optimizer(**t_opt_configs)
        )

        self.t_opt_configs = t_opt_configs
        self.decoder = decoder_network

        self.rnd_net = rnd_network()
        self.rnd_net_params = FrozenDict(self.rnd_net.init(rnd_key, jnp.ones((1, 12))).pop('params'))
        self.rnd_net_trainState = None

        self.ext_coeff = ext_coeff
        self.int_coeff = int_coeff
        self.rnd_rate = rnd_rate
    
    def start_task(self, task_id: int, description: str):
        super().start_task(task_id, description)

        actor_params = self.actor.params
        task_embedding = []
        for k in actor_params.keys():
            if k.startswith('embeds'):
                task_embedding.append(actor_params[k]['embedding'][task_id])
        task_embedding_alpha = jnp.stack(task_embedding, axis=0)
        task_embedding_mask = ste_step_fn(task_embedding_alpha)
        task_embedding_mask = jnp.expand_dims(task_embedding_mask, axis=[0, 3])
        self.rnd_net.task_embedding_mask = task_embedding_mask
        self.rnd_net_trainState = MPNTrainState.create(
            apply_fn=self.rnd_net.apply,
            params=self.rnd_net_params,
            tx=utils_fn.set_optimizer(**self.t_opt_configs)
        )
    
    def update(self, task_id: int, batch: Batch, optimize_alpha: bool = False) -> utils_fn.Dict[str, float]:
        if not self.update_coef:
            optimize_alpha = False
            
        update_target = self.step % self.target_update_period == 0

        new_rng, new_actor, new_temp, new_critic, new_target_critic, info, new_decoder = _update_cotasp_jit(
            self.rng, task_id, self.tau, self.discount, self.target_entropy, optimize_alpha, 
            self.param_masks, self.actor, self.critic, self.target_critic, update_target,
            self.temp, batch, self.decoder, self.rnd_net_trainState, self.ext_coeff, self.int_coeff, self.rnd_rate
        )

        self.step += 1
        self.rng = new_rng
        self.actor = new_actor
        self.temp = new_temp  
        self.critic = new_critic
        self.target_critic = new_target_critic   
        self.decoder = new_decoder

        return info

def _update_critic(
    rng: PRNGKey, task_id: int, actor: MPNTrainState, critic: TrainState, 
    target_critic: TrainState, update_target: bool, temp: TrainState, batch: Batch, 
    discount: float, tau: float, decoder: TrainState, rnd_net: TrainState, ext_coeff: float, int_coeff: float) -> Tuple[PRNGKey, TrainState, TrainState, InfoDict]:
    
    rng, key = jax.random.split(rng)
    dist, dicts = actor(batch.next_observations, jnp.array([task_id]))
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)
    next_q -= temp() * next_log_probs

    # >>>>>>>>>>>>>>>> add intrisic reward >>>>>>>>>>>>>>>>>>>>>>>
    ext_reward = batch.rewards # task reward
    encoder_output = dicts['encoder_output']
    pre_input = jnp.concatenate([encoder_output, batch.actions], -1)
    predict_z = decoder.apply_fn({'params': decoder.params}, pre_input)
    target_z = rnd_net.apply_fn({'params': rnd_net.params}, batch.next_observations)
    @jax.vmap
    def vector_norm(x): # L2 norm
        return jnp.sqrt(jnp.sum(jnp.square(x)))
    intrisic_reward = vector_norm(predict_z - target_z)

    reward = ext_coeff * ext_reward + int_coeff * intrisic_reward
    # <<<<<<<<<<<<<<<< add intrisic reward <<<<<<<<<<<<<<<<<<<<<<<

    target_q = reward + discount * batch.masks * next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()}
    
    grads_critic, critic_info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    # recording info
    critic_info['g_norm_critic'] = global_norm(grads_critic)

    new_critic = critic.apply_gradients(grads=grads_critic)

    # >>>>>>>>>>>>>>>>>>>>>> recording reward bonus info >>>>>>>>>>>>>>>>>>>>>>>>>>
    critic_info['int_reward/mean'] = jnp.mean(intrisic_reward)
    critic_info['int_reward/max'] = jnp.max(intrisic_reward)
    critic_info['int_reward/min'] = jnp.min(intrisic_reward)
    critic_info['int_reward/std'] = jnp.std(intrisic_reward)
    critic_info['ext_reward/mean'] = jnp.mean(ext_reward)
    critic_info['ext_reward/max'] = jnp.max(ext_reward)
    critic_info['ext_reward/min'] = jnp.min(ext_reward)
    critic_info['ext_reward/std'] = jnp.std(ext_reward)
    # <<<<<<<<<<<<<<<<<<<<<< recording reward bonus info <<<<<<<<<<<<<<<<<<<<<<<<<<
    
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_update(new_critic, target_critic, 0)

    return rng, new_critic, new_target_critic, critic_info   

        