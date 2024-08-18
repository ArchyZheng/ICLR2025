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
            nn.Conv(features=1, kernel_size=(4, 4), strides=(1, 1), name='conv1', padding='VALID'),
            activation_fn('relu'),])
        self.mlp = [nn.Dense(features=hidn, name=f'mlp_{i}') for i, hidn in enumerate(self.mlp_features)]

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
        # CNN setup
        self.rnd_cnn = RND_CNN()
        # MLP setup
        self.mlp_obs = nn.Sequential([
            nn.Dense(features=256, name='fc1'),
            nn.LayerNorm(),
            activation_fn('leaky_relu'),
            nn.Dense(features=64, name='fc3')])
        
        self.final_output = nn.Sequential([
            nn.Dense(features=64, name='fc4')])

    def __call__(self, 
                 x: jnp.ndarray, task_mask: jnp.ndarray):
        
        mask_process = self.rnd_cnn(task_mask)
        mask_output_reshape = jnp.tile(mask_process, (x.shape[0], 1))
        phi_next_st = self.mlp_obs(x)
        target_next_st = jnp.multiply(phi_next_st, mask_output_reshape)
        phi_next_st = self.final_output(target_next_st)
        return phi_next_st
    