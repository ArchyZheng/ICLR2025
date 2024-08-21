# # %%
# import os
# import sys
# rootdir = os.path.join(os.getcwd(), '../../')
# sys.path.append(rootdir)

# %%
from jax.numpy import ndarray
import flax.linen as nn
import jax.numpy as jnp
import jax
from jaxrl.networks.common import default_init, activation_fn
from flax.core import FrozenDict
from jax import custom_jvp
from jaxrl.networks.policies_PRE import Decoder_PRE

class RND_CNN(nn.Module):
    mlp_features = [1016]
    def setup(self):
        self.cnn = nn.Sequential([
            nn.Conv(features=1, kernel_size=(4, 4), strides=(1, 1), name='conv1', kernel_init=default_init(jnp.sqrt(2)), padding='VALID'),
            activation_fn('leaky_relu'),])
        self.mlp = nn.Dense(features=1016, name=f'mlp_1')

    def __call__(self, x):
        x = self.cnn(x)
        x = jnp.reshape(x, (x.shape[0], -1)) # flatten
        x = self.mlp(x)
        x = activation_fn('leaky_relu')(x)
        return x

class rnd_network(nn.Module):
    """
    This is created by Chengqi. 
    Input the features
    Output the x. 
    """

    def setup(self):
        self.rnd_cnn = RND_CNN()
        
        # MLP setup
        self.mlp_obs = nn.Sequential([
            nn.Dense(features=256, name='fc1'),
            activation_fn('relu'),
            nn.Dense(features=256, name='fc2'),
            activation_fn('relu'),
            nn.Dense(features=256, name='fc3'),
            activation_fn('relu'),
            nn.Dense(features=32, name='fc4')])

    def __call__(self, 
                 x: jnp.ndarray, task_mask: jnp.ndarray):
        
        mask_process = self.rnd_cnn(task_mask)
        mask_output_reshape = jnp.tile(mask_process, (x.shape[0], 1))
        target_next_st = jnp.concatenate([x, mask_output_reshape], -1)
        phi_next_st = self.mlp_obs(target_next_st)

        return phi_next_st

class RND(nn.Module):
    
    def setup(self):
        self.target_network = rnd_network()
        self.predict_network = Decoder_PRE()
    
    def __call__(self, next_observations: jnp.ndarray, task_mask: jnp.ndarray, observations: jnp.ndarray, actions: jnp.ndarray):
        target = self.target_network(next_observations, task_mask)
        pred = self.predict_network(jnp.concatenate([observations, actions], -1))
        return pred, jax.lax.stop_gradient(target)