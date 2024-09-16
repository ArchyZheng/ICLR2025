# %%
import jax.numpy as jnp
from jax import random, grad, jit
import matplotlib.pyplot as plt

# Generate training data
x_train = jnp.linspace(-jnp.pi, jnp.pi, 100).reshape(-1, 1)
y_train = jnp.sin(x_train)

# Initialize parameters for a simple feedforward neural network
def init_params(key):
    params = {
        'W1': random.normal(key, (1, 32)),
        'b1': jnp.zeros((32,)),
        'W2': random.normal(key, (32, 32)),
        'b2': jnp.zeros((32,)),
        'W3': random.normal(key, (32, 1)),
        'b3': jnp.zeros((1,)),
    }
    return params

# Define a simple neural network
def forward(params, x):
    h1 = jnp.tanh(jnp.dot(x, params['W1']) + params['b1'])
    h2 = jnp.tanh(jnp.dot(h1, params['W2']) + params['b2'])
    y_pred = jnp.dot(h2, params['W3']) + params['b3']
    return y_pred

# Mean squared error loss function
def loss(params, x, y):
    y_pred = forward(params, x)
    return jnp.mean((y_pred - y) ** 2)

# Gradient descent update function
def update(params, x, y, learning_rate=0.001):
    grads = grad(loss)(params, x, y)
    return {key: params[key] - learning_rate * grads[key] for key in params}

# Training loop
key = random.PRNGKey(0)
params = init_params(key)
losses = []

for epoch in range(5000):
    params = update(params, x_train, y_train)
    if epoch % 100 == 0:
        current_loss = loss(params, x_train, y_train)
        losses.append(current_loss)

# Predict on test data
x_test = jnp.linspace(-jnp.pi, jnp.pi, 100).reshape(-1, 1)
y_pred = forward(params, x_test)

# Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot the sine prediction
ax1.plot(x_test, jnp.sin(x_test), label='True Sine')
ax1.plot(x_test, y_pred, label='Predicted Sine')
ax1.legend()
ax1.set_title('Sine Prediction')

# Plot the loss
ax2.plot(range(0, 5000, 100), losses)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss')

plt.tight_layout()
plt.show()

# %%
