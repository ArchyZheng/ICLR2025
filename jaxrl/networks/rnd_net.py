from jax.numpy import ndarray
from jaxrl.networks.policies import MetaPolicy
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
import jax
from jaxrl.networks.common import default_init, activation_fn
from flax.core import FrozenDict
from jaxrl.networks.common import TrainState
import jaxrl.networks.common as utils_fn
from jax import custom_jvp
from jaxrl.networks.common import InfoDict, TrainState, PRNGKey, Params, \
    MPNTrainState
class RND_CNN(nn.Module):
    mlp_features = [128, 64]
    def setup(self):
        self.cnn = nn.Sequential([
            nn.Conv(features=16, kernel_size=(8, 8), strides=(4, 4), name='conv1', kernel_init=default_init(jnp.sqrt(2))),
            activation_fn('relu'),
            nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2), name='conv2', kernel_init=default_init(jnp.sqrt(2))),
            activation_fn('relu')])
        self.mlp = [nn.Dense(features=hidn, name=f'mlp_{i}') for i, hidn in enumerate(self.mlp_features)]

    @nn.compact
    def __call__(self, x):
        x = self.cnn(x)
        x = jnp.reshape(x, (x.shape[0], -1)) # flatten
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i < len(self.mlp) - 1:
                x = activation_fn('leaky_relu')(x)
        return x

class rnd_network(nn.Module):
    """
    This is created by Chengqi. 
    Input the features
    Output the x. 
    """

    def setup(self):
        self.cnn_key = jax.random.PRNGKey(110)
        self.task_num = 10
        # CNN setup
        self.rnd_cnn = RND_CNN()
        self.rnd_cnn_params = FrozenDict({'params': self.rnd_cnn.init(self.cnn_key, jnp.ones((1, 4, 1024, 1)))['params']})
        
        # MLP setup
        self.mlp_obs = nn.Sequential([
            nn.Dense(features=256, name='fc1'),
            nn.LayerNorm(),
            activation_fn('leaky_relu'),
            nn.Dense(features=64, name='fc3')])
        
        self.final_output = nn.Sequential([
            nn.Dense(features=64, name='fc4')])

    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray, task_mask: jnp.ndarray):
        
        mask_process = self.rnd_cnn.apply({'params': self.rnd_cnn_params['params']}, task_mask)
        mask_output_reshape = jnp.tile(mask_process, (x.shape[0], 1))
        phi_next_st = self.mlp_obs(x)
        target_next_st = jnp.multiply(phi_next_st, mask_output_reshape)
        phi_next_st = self.final_output(target_next_st)

        return phi_next_st
    