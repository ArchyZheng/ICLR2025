# %%
import jax
import flax
import jax.numpy as jnp
# %%
import flax.linen as nn

class mlp_1024(nn.Module):
    features = [1024, 1024, 1024, 1024]
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.features)):
            x = nn.Dense(self.features[i])(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class mlp_256(nn.Module):
    features = [256, 256]
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.features)):
            x = nn.Dense(self.features[i])(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# %%
from flax.training.train_state import TrainState
import time
import optax
mlp_1024_def = mlp_1024()
mlp_1024_params = mlp_1024_def.init(jax.random.PRNGKey(0), jnp.ones((1, 12)))
mlp_1024_state = TrainState.create(apply_fn=mlp_1024_def.apply, params=mlp_1024_params, tx=optax.adam(1e-3))

mlp_256_def = mlp_256()
mlp_256_params = mlp_256_def.init(jax.random.PRNGKey(0), jnp.ones((1, 12)))
mlp_256_state = TrainState.create(apply_fn=mlp_256_def.apply, params=mlp_256_params, tx=optax.adam(1e-3))
# %%
start_time = time.time()
for i in range(1000):
    mlp_1024_state.apply_fn(mlp_1024_state.params, jnp.ones((1, 12)))
print('Time for mlp_1024:', time.time() - start_time)


start_time = time.time()
for i in range(1000):
    mlp_256_state.apply_fn(mlp_256_state.params, jnp.ones((1, 12)))
print('Time for mlp_256:', time.time() - start_time)

# %%
def loss(params):
    return jnp.sum(mlp_1024_state.apply_fn(params, jnp.ones((1, 12))))

def loss_1(params):
    return jnp.sum(mlp_256_state.apply_fn(params, jnp.ones((1, 12))))

grad_fn = jax.grad(loss)
start_time = time.time()
value_and_grad_fn = jax.value_and_grad(loss)
value_and_grad_fn_1 = jax.value_and_grad(loss_1)
for i in range(1000):
    value, grad = value_and_grad_fn(mlp_1024_state.params)
    mlp_1024_state = mlp_1024_state.apply_gradients(grads=grad)
print('Time for mlp_1024:', time.time() - start_time)


start_time = time.time()
for i in range(1000):
    value, grad_1 = value_and_grad_fn_1(mlp_256_state.params)
    mlp_256_state = mlp_256_state.apply_gradients(grads=grad_1)
print('Time for mlp_256:', time.time() - start_time)
# %%
