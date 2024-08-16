from jax.numpy import ndarray
import jax.numpy as jnp
import functools
from jaxrl.agents.sac.sac_learner import CoTASPLearner
from jaxrl.agents.sac.FGMaskLearner import MaskCombinationLearner
# from sac_learner import CoTASPLearner
from typing import Any, Sequence, Tuple
import numpy as np
from jaxrl.datasets.dataset import Batch
from jaxrl.dict_learning.task_dict import OnlineDictLearnerV2
from jaxrl.agents.sac.critic import target_update
from flax.core import freeze, unfreeze, FrozenDict
from jaxrl.networks.common import InfoDict, Params, PRNGKey, MPNTrainState, TrainState
import jaxrl.networks.common as utils_fn
import jax
from flax import linen as nn
from optax import global_norm
from jaxrl.networks.common import default_init, activation_fn
class PredictNet(nn.Module):
    features = [128, 64]
    active_fn: Any = nn.leaky_relu

    def setup(self) -> None:
        self.backbones = [nn.Dense(feat) for feat in self.features]
    @nn.compact
    def __call__(self, x: ndarray) -> ndarray:
        for i, backbone in enumerate(self.backbones):
            x = backbone(x)
            if i < len(self.backbones) - 1:
                x = self.active_fn(x)
        return x

@functools.partial(jax.jit, static_argnames=('update_target'))
def _update_cotasp_jit(rng: PRNGKey, task_id: int, tau: float, discount: float, 
    target_entropy: float, optimize_alpha: bool, param_mask: FrozenDict[str, Any], 
    actor: MPNTrainState, critic: TrainState, target_critic: TrainState, 
    update_target, temp: TrainState, batch: Batch
    ) -> Tuple[PRNGKey, MPNTrainState, TrainState, TrainState, TrainState, InfoDict]:
    # optimizing critics 
    new_rng, new_critic, new_target_critic, critic_info = _update_critic(
        rng, task_id, actor, critic, target_critic, update_target, temp, 
        batch, discount, tau
    )

    # optimizing either alpha or theta
    # new_rng, new_actor, actor_info = jax.lax.cond(
    #     optimize_alpha,
    #     _update_alpha,
    #     _update_theta,
    #     new_rng, task_id, param_mask, actor, new_critic, temp, batch
    # )
    new_rng, new_actor, actor_info = _update_theta(new_rng, task_id, param_mask, actor, new_critic, temp, batch)

    # updating temperature coefficient
    new_temp, temp_info = _update_temp(
        temp, actor_info['entropy'], target_entropy
    )

    return new_rng, new_actor, new_temp, new_critic, new_target_critic, {
        **actor_info,
        **temp_info,
        **critic_info
    }

class TargetNet(nn.Module):
    features = [128, 64]
    mask_output_features = [32, 64]
    combine_features = [64, 64]
    active_fn: Any = nn.leaky_relu

    def setup(self) -> None:
        # mask part:
        self.mask_cnn = nn.Sequential([
            nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name='conv1', kernel_init=default_init(jnp.sqrt(2))),
            activation_fn('relu'),
            nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name='conv2', kernel_init=default_init(jnp.sqrt(2))),
            activation_fn('relu'),
            nn.Conv(features=64, kernel_size=(4, 4), strides=(1, 1), name='conv3', kernel_init=default_init(jnp.sqrt(2))),
            activation_fn('relu')])
        self.mask_output = [nn.Dense(feat) for feat in self.mask_output_features]

        # state part:
        self.next_obs_output = [nn.Dense(feat) for feat in self.features]

        # combine part:
        self.combine_output = [nn.Dense(feat) for feat in self.combine_features]
    @nn.compact
    def __call__(self, x: ndarray, task_mask: ndarray) -> ndarray:
        # mask part: mask (1, 4, 1024, 1)
        mask_output = self.mask_cnn(task_mask)
        mask_output = mask_output.reshape((mask_output.shape[0], -1)) # flatten
        for i, layer in enumerate(self.mask_output):
            mask_output = layer(mask_output)
            if i < len(self.mask_output) - 1:
                mask_output = self.active_fn(mask_output)
        
        # state encoder:
        for i, backbone in enumerate(self.backbones):
            x = backbone(x)
            if i < len(self.backbones) - 1:
                x = self.active_fn(x)
        
        # combine part:
        combine = jnp.multiply(x, mask_output)
        for i, backbone in enumerate(self.combine_output):
            combine = backbone(combine)
            if i < len(self.combine_output) - 1:
                combine = self.active_fn(combine)
        return combine

class RNDLearner(MaskCombinationLearner):
    def __init__(self, seed: int, observations: ndarray, actions: ndarray, task_num: int, load_policy_dir: str | None = None, load_dict_dir: str | None = None, update_dict=True, update_coef=True, dict_configs_fixed: dict = ..., dict_configs_random: dict = ..., pi_opt_configs: dict = ..., q_opt_configs: dict = ..., t_opt_configs: dict = ..., actor_configs: dict = ..., critic_configs: dict = ..., tau: float = 0.005, discount: float = 0.99, target_update_period: int = 1, target_entropy: float | None = None, init_temperature: float = 1):
        super().__init__(seed, observations, actions, task_num, load_policy_dir, load_dict_dir, update_dict, update_coef, dict_configs_fixed, dict_configs_random, pi_opt_configs, q_opt_configs, t_opt_configs, actor_configs, critic_configs, tau, discount, target_update_period, target_entropy, init_temperature)
        self.rng, key = jax.random.split(self.rng)

        
        # <<<<<<<<<<<<<<< RND part <<<<<<<<<<<<<<<<
        # predictor part
        predict_net = PredictNet()
        dummy_input = jnp.append(jnp.ones((1, 1024)), actions, axis=-1)
        predict_params = FrozenDict(predict_net.init(key, dummy_input).pop('params'))
        self.predictor = TrainState.create(
            apply_fn=predict_net.apply,
            params=predict_params,
            tx=utils_fn.set_optimizer(**t_opt_configs)
        )

        # target part
        target_net = TargetNet()
        dummy_task_mask = jnp.zeros((1, 4, 1024, 1))
        target_params = FrozenDict(target_net.init(key, observations, dummy_task_mask).pop('params'))
        self.target = TrainState.create(
            apply_fn=target_net.apply,
            params=target_params,
            tx=utils_fn.set_optimizer(**t_opt_configs)
        )   
        # >>>>>>>>>>>>>>> RND part >>>>>>>>>>>>>>>>
    
    def update(self, task_id: int, batch: Batch, optimize_alpha: bool=False) -> InfoDict:

        if not self.update_coef:
            optimize_alpha = False
            
        update_target = self.step % self.target_update_period == 0

        new_rng, new_actor, new_temp, new_critic, new_target_critic, info = _update_cotasp_jit(
            self.rng, task_id, self.tau, self.discount, self.target_entropy, optimize_alpha, 
            self.param_masks, self.actor, self.critic, self.target_critic, update_target,
            self.temp, batch
        )

        self.step += 1
        self.rng = new_rng
        self.actor = new_actor
        self.temp = new_temp  
        self.critic = new_critic
        self.target_critic = new_target_critic   

        return info
    
def _update_predictor(predictor: TrainState, target: TrainState, encoder_output: jnp.ndarray, batch, task_mask):
    info = {}
    def loss_fn(predictor_params: Params) -> ndarray:
        predictor_input = jnp.append(encoder_output, batch.actions, axis=-1)
        predict_z = predictor(predictor_input)
        target_z = target(predictor_input, task_mask)
        rnd_loss = jnp.mean(jnp.square(predict_z - target_z))
        return rnd_loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(predictor.params)
    info['rnd_loss'] = loss
    new_predictor = predictor.apply_gradients(grads=grad)

    return predictor, info

def _update_critic(
    rng: PRNGKey, task_id: int, actor: MPNTrainState, critic: TrainState, 
    target_critic: TrainState, update_target: bool, temp: TrainState, batch: Batch, 
    discount: float, tau: float) -> Tuple[PRNGKey, TrainState, TrainState, InfoDict]:
    
    rng, key = jax.random.split(rng)
    dist, _ = actor(batch.next_observations, jnp.array([task_id]))
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)
    next_q -= temp() * next_log_probs
    target_q = batch.rewards + discount * batch.masks * next_q

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
    
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_update(new_critic, target_critic, 0)

    return rng, new_critic, new_target_critic, critic_info

def _update_theta(
    rng: PRNGKey, task_id: int, param_mask: FrozenDict[str, Any], 
    actor: MPNTrainState, critic: TrainState, temp: TrainState, 
    batch: Batch) -> Tuple[PRNGKey, MPNTrainState, InfoDict]:

    rng, key = jax.random.split(rng)
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, dicts = actor.apply_fn(
            {'params': actor_params}, batch.observations, jnp.array([task_id])
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()

        _info = {
            'hac_sac_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'means': dicts['means'].mean()
        }
        for k in dicts['masks']:
            _info[k+'_rate_act'] = jnp.mean(dicts['masks'][k])

        return actor_loss, _info
    
    # grads of actor
    grads_actor, actor_info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
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

    return rng, new_actor, actor_info

def _update_temp(
    temp: TrainState, actor_entropy: float, target_entropy: float
    ) -> Tuple[TrainState, InfoDict]:

    def temperature_loss_fn(temp_params: Params):
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (actor_entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}
    
    grads_temp, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(temp.params)
    # recording info
    temp_info['g_norm_temp'] = global_norm(grads_temp)

    new_temp = temp.apply_gradients(grads=grads_temp)

    return new_temp, temp_info
