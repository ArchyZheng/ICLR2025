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
    mlp_features = [512, 512, 256]
    def setup(self):
        self.cnn = nn.Sequential([
            nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name='conv1', kernel_init=default_init(jnp.sqrt(2))),
            activation_fn('relu'),
            nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name='conv2', kernel_init=default_init(jnp.sqrt(2))),
            activation_fn('relu'),
            nn.Conv(features=64, kernel_size=(4, 4), strides=(1, 1), name='conv3', kernel_init=default_init(jnp.sqrt(2))),
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
        self.rnd_cnn_params = FrozenDict(self.rnd_cnn.init(self.cnn_key, jnp.ones((1, 4, 1024, 1))).pop('params')) # was [10, 256, 1024]
        # MLP setup
        self.mlp_obs = nn.Sequential([
            nn.Dense(features=256, name='fc1'),
            nn.LayerNorm(),
            activation_fn('leaky_relu'),
            nn.Dense(features=256, name='fc2'),
            nn.LayerNorm(),
            activation_fn('leaky_relu'),
            nn.Dense(features=256, name='fc3')])
        self.mlp_output = nn.Sequential([
            nn.Dense(features=256, name='output_fc1'),
            nn.LayerNorm(),
            activation_fn('leaky_relu'),
            nn.Dense(features=64, name='output_fc2')])
        self.task_embedding_mask = jnp.ones((1, 4, 1024, 1))
        self.task_embedding_results = []

    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray,
                 task_id: jnp.ndarray):
        if len(self.task_embedding_results) == task_id:
            rnd_cnn_output = self.rnd_cnn.apply({'params': self.rnd_cnn_params}, self.task_embedding_mask)
            self.task_embedding_results.append(rnd_cnn_output)
        else:
            rnd_cnn_output = self.task_embedding_results[task_id]

        rnd_cnn_output_reshaped = jnp.tile(rnd_cnn_output, (x.shape[0], 1))
        phi_next_st = self.mlp_obs(x)
        target_next_st = jnp.multiply(phi_next_st, rnd_cnn_output_reshaped)
        target_next_st = self.mlp_output(target_next_st)
        return target_next_st