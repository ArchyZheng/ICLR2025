# # %%
# import os
# import sys
# rootdir = os.path.join(os.getcwd(), '../../')
# sys.path.append(rootdir)
# %%
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

    
class Decoder_PRE(nn.Module):
    """
    This is created by Chengqi. 
    Input the features
    Output the x. 
    """
    features = [1024, 256, 64]

    def setup(self):
        self.decoder = [nn.Dense(hidn, kernel_init=default_init()) for hidn in self.features]

    def __call__(self, x):
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(self.decoder) - 1:
                x = activation_fn('relu')(x)
        return x