# %%
import os 
os.environ["MUJOCO_GL"] = "egl"
from continual_world import TASK_SEQS, get_single_env
import numpy as np
import jax.numpy as jnp
import random
import matplotlib.pyplot as plt
from jaxrl.networks import policies
# %%
env_name = "cw1-hammer"
seed = 110
randomization = "random_init_all"
normalize_reward = True
seq_tasks = TASK_SEQS[env_name]
env = get_single_env(name=seq_tasks[0]['task'], seed=seed, randomization=randomization, normalize_reward=normalize_reward)
# %%
# plt.imsave(f"test.png", env.render(mode="rgb_array"))
# folder_name = "test_trajectory"
# for i in range(200):
#     action = env.action_space.sample()
#     next_observation, reward, done, info = env.step(action)
#     plt.imsave(f"{folder_name}/{i}.png", env.render(mode="rgb_array"))
# %% policy part
import jax
from flax.training.train_state import TrainState
from jaxrl.networks.common import MPNTrainState
import jaxrl.networks.common as utils_fn
import optax
from flax.core import freeze, unfreeze, FrozenDict
actor_configs = {
    "clip_mean": 1.0,
    "final_fc_init_scale": 1.0e-4,
    "hidden_dims": [1024, 1024, 1024, 1024],
    "name_activation": "leaky_relu",
    "state_dependent_std": True,
    "use_layer_norm": True,
    "action_dim": 4,
    "task_num": 10,
}
pi_opt_configs = {
  "optim_algo": "adam",
  "clip_method": None,
  "max_norm": -1,
  "opt_kargs": {
    "learning_rate": 3.0e-4}
}
# %%
actor_def = policies.MetaPolicy(**actor_configs)
actor_params = FrozenDict(actor_def.init(jax.random.PRNGKey(seed+1), jnp.zeros((1, env.observation_space.shape[0])), jnp.array([0])).pop('params'))
actor = MPNTrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=utils_fn.set_optimizer(**pi_opt_configs))
# load_policy_dir = "logs/saved_actors/cw1-stick-pull__cotasp__110__1725276631.json"
# load_policy_dir = "logs/saved_actors/cw1-stick-pull__cotasp__220__1725279907.json"
# load_policy_dir = "logs/saved_actors/cw1-stick-pull__cotasp__330__1725279915.json"
# load_policy_dir = "logs/saved_actors/cw1-stick-pull__cotasp__330__1725634114.json"
# load_policy_dir = "logs/saved_actors/cw1-stick-pull__cotasp__440__1725676451.json"
# load_policy_dir = "logs/saved_actors/cw1-stick-pull__cotasp__440__1725673321.json"
load_policy_dir = "actor_1.pkl"
# load_policy_dir = "actor.pkl"
actor = actor.load(load_policy_dir)
# %%
dummy_policy = lambda observation, actor=actor, task=jnp.array([1]): actor(x=observation, t=task)[0]
final_action = lambda observation, seed=jax.random.PRNGKey(seed): dummy_policy(observation=observation.reshape(1, -1)).sample(seed=seed)[0]
dummy_observation_0 = jnp.array([[-0.032652, 0.51487875, 0.23688607, -0.00787378, 0.6232033, 0.02, 0.2, 0.6, 0.08, 0.30898893, 0.45203695, 0.02]])
# %%
# delta_x = jnp.array([0.01] * 3)
delta_vector = lambda begin_idx, delt_o: jnp.array([delt_o if i in range(begin_idx, begin_idx + 3) else 0 for i in range(0, 12)])
change_observation = ["change gripper position", "change stick position", "change cup position", "change goal position"]
# %%
for i in range(4):
    begin_index = i * 3
    dummy_observation_2 = dummy_observation_0.copy() + delta_vector(begin_index, delt_o=0.01)
    print(f"{change_observation[i]}")
    delta_action = final_action(dummy_observation_0) - final_action(dummy_observation_2)
    print(jnp.sqrt(jnp.sum(delta_action ** 2)))
# %%
def trajectory_generator(env, policy, n_steps=200):
    res = (640, 480)
    camera = "topview"
    observation, done = env.reset(), False
    for index in range(n_steps):
        try:
            action = policy(observation)
        except:
            action = policy()
        observation, r, done, info = env.step(action)
        # print(f"observation: {observation}")
        # if info['success'] or done == 1.0:
        #     observation, done = env.reset(), False
        #     print(f"observation: {observation}")
        # Camera is one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
        yield r, done, info, env.sim.render(*res, mode='offscreen', camera_name=camera)[:,:,::-1]
# %%
import cv2
def writer_for(tag, fps, res):
    if not os.path.exists('movies'):
        os.mkdir('movies')
    return cv2.VideoWriter(
        f'movies/{tag}.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'),
        fps,
        res
    )
# %%
resolution = (640, 480)
writer = writer_for("trial", env.metadata['video.frames_per_second'], res=resolution)
policy = final_action
camera = "topview"
# %%
collection_reward = []
for _ in range(1):
    for r, done, info, img in trajectory_generator(env, policy):
        img = cv2.rotate(img, cv2.ROTATE_180)
        writer.write(img)
        collection_reward.append(r)
        # if info['success']:
        #     break
writer.release()
# %%
plt.plot(collection_reward)
plt.show()
# %%
