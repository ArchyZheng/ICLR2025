# %%
import jax
from typing import List
import random
import gym
import metaworld
import numpy as np
from gym.wrappers import TimeLimit

from jaxrl import wrappers
from brax.envs import env as _env

from brax.envs.wrappers import VectorGymWrapper, VectorWrapper, GymWrapper
# from gym.wrappers.autoreset impor

from jaxrl.wrappers.normalization import RescaleReward

def get_mt50() -> metaworld.MT50:
    saved_random_state = np.random.get_state()
    np.random.seed(999)
    random.seed(999)
    MT50 = metaworld.MT50()
    np.random.set_state(saved_random_state)
    return MT50

TASK_SEQS = {
    "salina/halfcheetah/forgetting": [
        {'task': "hugefoot", 'hint': 'halfcheetah with hugefoot'},
        {'task': "moon", 'hint': 'halfcheetah in a small gravity environment'},
        {'task': "carry_stuff", 'hint': 'halfcheetah is carrying some stuff'},
        {'task': "rainfall", 'hint': 'halfcheetah is in a rainfall environment'},
    ],
    "salina/halfcheetah/transfer": [
        {'task': "carry_stuff_hugegravity", 'hint': 'halfcheetah with hugefoot'},
        {'task': "moon", 'hint': 'halfcheetah in a small gravity environment'},
        {'task': "defective_module", 'hint': 'halfcheetah is carrying some stuff'},
        {'task': "hugefoot_rainfall", 'hint': 'halfcheetah is in a rainfall environment'},
    ],
    "salina/halfcheetah/compositionality": [
        {'task': "tinyfoot", 'hint': 'halfcheetah with hugefoot'},
        {'task': "moon", 'hint': 'halfcheetah in a small gravity environment'},
        {'task': "carry_stuff_hugegravity", 'hint': 'halfcheetah is carrying some stuff'},
        {'task': "tinyfoot_moon", 'hint': 'halfcheetah is in a rainfall environment'},
    ],
    "salina/halfcheetah/robustness": [
        {'task': "normal", 'hint': 'halfcheetah with hugefoot'},
        {'task': "inverted_actions", 'hint': 'halfcheetah in a small gravity environment'},
        {'task': "normal", 'hint': 'halfcheetah is carrying some stuff'},
        {'task': "inverted_actions", 'hint': 'halfcheetah is in a rainfall environment'},
    ],
    "cw10": [
        {'task': "hammer-v1", 'hint': 'Hammer a screw on the wall.'},
        {'task': "push-wall-v1", 'hint': 'Bypass a wall and push a puck to a goal.'},
        {'task': "faucet-close-v1", 'hint': 'Rotate the faucet clockwise.'},
        {'task': "push-back-v1", 'hint': 'Pull a puck to a goal.'},
        {'task': "stick-pull-v1", 'hint': 'Grasp a stick and pull a box with the stick.'},
        {'task': "handle-press-side-v1", 'hint': 'Press a handle down sideways.'},
        {'task': "push-v1", 'hint': 'Push the puck to a goal.'},
        {'task': "shelf-place-v1", 'hint': 'Pick and place a puck onto a shelf.'},
        {'task': "window-close-v1", 'hint': 'Push and close a window.'},
        {'task': "peg-unplug-side-v1", 'hint': 'Unplug a peg sideways.'},
    ],
    "cw1-stick-pull": [
        {'task': "stick-pull-v1", 'hint': 'Grasp a stick and pull a box with the stick.'},
    ],
    "cw1-hammer": [
        {'task': "hammer-v1", 'hint': 'Hammer a screw on the wall.'},
    ],
    "cw1-push-back": [
        "push-back-v1"
    ],
    "cw1-push": [
        "push-v1"
    ],
    "cw2-test": [
        {'task': "push-wall-v1", 'hint': 'Bypass a wall and push a puck to a goal.'},
        {'task': "hammer-v1", 'hint': 'Hammer a screw on the wall.'},
    ],
    "cw2-ab-coffee-button": [
        {'task': "hammer-v1", 'hint': 'Hammer a screw on the wall.'},
        {'task': "coffee-button-v1", 'hint': 'Push a button on the coffee machine.'}
    ],
    "cw2-ab-handle-press": [
        {'task': "hammer-v1", 'hint': 'Hammer a screw on the wall.'},
        {'task': "handle-press-v1", 'hint': 'Press a handle down.'}
    ],
    "cw2-ab-window-open": [
        {'task': "hammer-v1", 'hint': 'Hammer a screw on the wall.'},
        {'task': "window-open-v1", 'hint': 'Push and open a window.'}
    ],
    "cw2-ab-reach": [
        {'task': "hammer-v1", 'hint': 'Hammer a screw on the wall.'},
        {'task': "reach-v1", 'hint': 'Reach a goal position.'}
    ],
    "cw2-ab-button-press": [
        {'task': "hammer-v1", 'hint': 'Hammer a screw on the wall.'},
        {'task': "button-press-v1", 'hint': 'Press a button.'}
    ],
    "cw3-test": [
        {'task': "push-back-v1", 'hint': 'Pull a puck to a goal.'},
        {'task': "shelf-place-v1", 'hint': 'Pick and place a puck onto a shelf.'},
        {'task': "stick-pull-v1", 'hint': 'Grasp a stick and pull a box with the stick.'},
    ]
}

TASK_SEQS["cw20"] = TASK_SEQS["cw10"] + TASK_SEQS["cw10"]
META_WORLD_TIME_HORIZON = 200
MT50 = get_mt50()

class RandomizationWrapper(gym.Wrapper):
    """Manages randomization settings in MetaWorld environments."""

    ALLOWED_KINDS = [
        "deterministic",
        "random_init_all",
        "random_init_fixed20",
        "random_init_small_box",
    ]

    def __init__(self, env: gym.Env, subtasks: List[metaworld.Task], kind: str) -> None:
        assert kind in RandomizationWrapper.ALLOWED_KINDS
        super().__init__(env)
        self.subtasks = subtasks
        self.kind = kind

        env.set_task(subtasks[0])
        if kind == "random_init_all":
            env._freeze_rand_vec = False
            env.seeded_rand_vec = True

        if kind == "random_init_fixed20":
            assert len(subtasks) >= 20

        if kind == "random_init_small_box":
            diff = env._random_reset_space.high - env._random_reset_space.low
            self.reset_space_low = env._random_reset_space.low + 0.45 * diff
            self.reset_space_high = env._random_reset_space.low + 0.55 * diff

    def reset(self, **kwargs) -> np.ndarray:
        if self.kind == "random_init_fixed20":
            self.env.set_task(self.subtasks[random.randint(0, 19)])
        elif self.kind == "random_init_small_box":
            rand_vec = np.random.uniform(
                self.reset_space_low, self.reset_space_high, size=self.reset_space_low.size
            )
            self.env._last_rand_vec = rand_vec

        return self.env.reset(**kwargs)


def get_subtasks(name: str) -> List[metaworld.Task]:
    return [s for s in MT50.train_tasks if s.env_name == name]

from brax.envs import Env as BraxEnv
def get_single_env(
    name, seed, 
    randomization="random_init_all",
    add_episode_monitor=True,
    normalize_reward=False
    ):
    salina_list = env_tasks.keys()
    if name == "HalfCheetah-v3" or name == "Ant-v3":
        env = gym.make(name)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    # >>>>>>>>>>>>>>>> add salina halfcheetah >>>>>>>>>>>>>>>
    elif name in salina_list:
        # env: BraxEnv = MyHalfcheetah(env_task=name)
        env: BraxEnv = Halfcheetah(env_task=name)
        env.seed = seed
        # env.batch_size = 10
        env = VectorWrapper(env, batch_size=10)
        env = GymWrapper(env, seed)
        # env = gym.vector.SyncVectorEnv([lambda: env for _ in range(10)])

        # env = VectorGymWrapper(env=env, seed=seed)
        # seed += 1000
        # env_list.append(env)
            
        # env.seed(seed)
        return env
    # <<<<<<<<<<<<<<<< add salina halfcheetah <<<<<<<<<<<<<<<
    else:
        env = MT50.train_classes[name]()
        env.seed(seed)
        env = RandomizationWrapper(env, get_subtasks(name), randomization)
        env.name = name
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env = TimeLimit(env, META_WORLD_TIME_HORIZON)
    env = gym.wrappers.ClipAction(env)
    if normalize_reward:
        env = RescaleReward(env, reward_scale=1.0 / META_WORLD_TIME_HORIZON)
    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)
    return env
# %%    

# if __name__ == "__main__":
#     import time

#     # def print_reward(env: gym.Env):
#     #     obs, done = env.reset(), False
#     #     i = 0
#     #     while not done:
#     #         i += 1
#     #         next_obs, rew, done, _ = env.step(env.action_space.sample())
#     #         print(i, rew)

#     # tasks_list = TASK_SEQS["cw1-push"]
#     # env = get_single_env(tasks_list[0], 1, "deterministic", normalize_reward=False)
#     # env_normalized = get_single_env(tasks_list[0], 1, "deterministic", normalize_reward=True)

#     # print_reward(env)
#     # print_reward(env_normalized)

#     tasks_list = TASK_SEQS["cw1-push"]
#     s = time.time()
#     env = get_single_env(tasks_list[0], 1, "random_init_all")
#     print(time.time() - s)
#     s = time.time()
#     env = get_single_env(tasks_list[0], 1, "random_init_all")
#     print(time.time() - s)

#     o = env.reset()
#     _, _, _, _ = env.step(np.array([np.nan, 1.0, -1.0, 0.0]))
#     o_new = env.reset()
#     print(o)
#     print(o_new)
# %%
# import brax.v1 as brax
import numpy as np
# from brax.v1 import jumpy as jp
from brax import jumpy as jp

# import numpy as jp
# from brax.envs.half_cheetah import Halfcheetah
# from brax.v1.envs.half_cheetah import Halfcheetah, _SYSTEM_CONFIG
import brax
from brax.envs.half_cheetah import Halfcheetah, _SYSTEM_CONFIG
from google.protobuf import text_format
#%%
OBS_DIM = 18
ACT_DIM = 6


class Halfcheetah(Halfcheetah):
    def __init__(self, env_task: str, **kwargs) -> None:
        self._forward_reward_weight = 1.0
        self._ctrl_cost_weight = 0.1
        self._reset_noise_scale = 0.1
        self._exclude_current_positions_from_observation = (True)
        config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
        env_specs = env_tasks[env_task]
        self.obs_mask = jp.concatenate(np.ones((1,OBS_DIM)))
        self.action_mask = jp.concatenate(np.ones((1,ACT_DIM)))
        self.seed = None
        self.name = env_task
        self.step_number = 0
        for spec,coeff in env_specs.items():
            if spec == "gravity":
                config.gravity.z *= coeff
            elif spec == "friction":
                config.friction *= coeff
            elif spec == "obs_mask":
                zeros = int(coeff * OBS_DIM)
                ones = OBS_DIM - zeros
                np.random.seed(0)
                self.obs_mask = jp.concatenate(np.random.permutation(([0]*zeros)+([1]*ones)).reshape(1,-1))
            elif spec == "action_mask":
                self.action_mask[coeff] = 0.
            elif spec == "action_swap":
                self.action_mask[coeff] = -1.
            else:
                for body in config.bodies:
                    if spec in body.name:
                        body.mass *= coeff
                        body.colliders[0].capsule.radius *= coeff
        self.sys = brax.System(config)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Observe halfcheetah body position and velocities."""
        joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)
        # qpos: position and orientation of the torso and the joint angles
        # TODO: convert rot to just y-ang component
        if self._exclude_current_positions_from_observation:
            qpos = [qp.pos[0, 2:], qp.rot[0, (0, 2)], joint_angle]
        else:
            qpos = [qp.pos[0, (0, 2)], qp.rot[0, (0, 2)], joint_angle]
        # qvel: velocity of the torso and the joint angle velocities
        qvel = [qp.vel[0, (0, 2)], qp.ang[0, 1:2], joint_vel]
        return jp.concatenate(qpos + qvel) * self.obs_mask

    def reset(self, rng: jp.ndarray) -> _env.State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jp.random_split(rng, 3)

        qpos = self.sys.default_angle() + self._noise(rng1)
        qvel = self._noise(rng2)

        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        self._qps = [qp]
        obs = self._get_obs(qp, self.sys.info(qp))
        reward, done, zero = jp.zeros(3)
        metrics = {
            'x_position': zero,
            'x_velocity': zero,
            'reward_ctrl': zero,
            'reward_run': zero,
        }
        return _env.State(qp, obs, reward, done, metrics)
        

    def step(self, state: _env.State, action: jp.ndarray) -> _env.State:
        """Run one timestep of the environment's dynamics."""
        action = action * self.action_mask
        qp, info = self.sys.step(state.qp, action)
        self._qps.append(qp)

        velocity = (qp.pos[0] - state.qp.pos[0]) / self.sys.config.dt
        forward_reward = self._forward_reward_weight * velocity[0]
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(qp, info)
        reward = forward_reward - ctrl_cost
        state.metrics.update(
            x_position=qp.pos[0, 0],
            x_velocity=velocity[0],
            reward_run=forward_reward,
            reward_ctrl=-ctrl_cost)
        return state.replace(qp=qp, obs=obs, reward=reward)

    def get_qps(self):
        return self._qps


class MyHalfcheetah(Halfcheetah):
    def __init__(self, env_task: str, **kwargs) -> None:
        self._forward_reward_weight = 1.0
        self._ctrl_cost_weight = 0.1
        self._reset_noise_scale = 0.1
        self._exclude_current_positions_from_observation = (True)
        config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
        env_specs = env_tasks[env_task]
        self.obs_mask = jp.concatenate(np.ones((1,OBS_DIM)))
        self.action_mask = jp.concatenate(np.ones((1,ACT_DIM)))
        self.seed = None
        self.name = env_task
        self.step_number = 0
        for spec,coeff in env_specs.items():
            if spec == "gravity":
                config.gravity.z *= coeff
            elif spec == "friction":
                config.friction *= coeff
            elif spec == "obs_mask":
                zeros = int(coeff * OBS_DIM)
                ones = OBS_DIM - zeros
                np.random.seed(0)
                self.obs_mask = jp.concatenate(np.random.permutation(([0]*zeros)+([1]*ones)).reshape(1,-1))
            elif spec == "action_mask":
                self.action_mask[coeff] = 0.
            elif spec == "action_swap":
                self.action_mask[coeff] = -1.
            else:
                for body in config.bodies:
                    if spec in body.name:
                        body.mass *= coeff
                        body.colliders[0].capsule.radius *= coeff
        self.sys = brax.System(config)
        self.step_temp = jax.jit(self.sys.step)

    def _get_obs(self, qp, info) -> jp.ndarray:
        """Observe halfcheetah body position and velocities."""
        joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)
        # qpos: position and orientation of the torso and the joint angles
        # TODO: convert rot to just y-ang component
        if self._exclude_current_positions_from_observation:
            qpos = [qp.pos[0, 2:], qp.rot[0, (0, 2)], joint_angle]
        else:
            qpos = [qp.pos[0, (0, 2)], qp.rot[0, (0, 2)], joint_angle]
        # qvel: velocity of the torso and the joint angle velocities
        qvel = [qp.vel[0, (0, 2)], qp.ang[0, 1:2], joint_vel]
        return jp.concatenate(qpos + qvel) * self.obs_mask

    def reset(self, rng = None):
        """Resets the environment to an initial state."""
        # if rng == None:
        rng = jp.random_prngkey(self.seed)
        self.seed = self.seed + 1
        rng, rng1, rng2 = jp.random_split(rng, 3)

        qpos = self.sys.default_angle() + self._noise(rng1)
        qvel = self._noise(rng2)

        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        self._qps = [qp]
        obs = self._get_obs(qp, self.sys.info(qp))
        reward, done, zero = jp.zeros(3)
        self.step_number = 0
        metrics = {
            'x_position': zero,
            'x_velocity': zero,
            'reward_ctrl': zero,
            'reward_run': zero,
        }
        self.state = _env.State(qp, obs, reward, done, metrics) 
        return self.state.obs

    def step(self, action: jp.ndarray, state = None):
        """Run one timestep of the environment's dynamics."""
        if state == None:
            state = self.state
        action = action * self.action_mask
        self.step_number += 1
        qp, info = self.step_temp(state.qp, action)
        self._qps.append(qp)

        velocity = (qp.pos[0] - state.qp.pos[0]) / self.sys.config.dt
        forward_reward = self._forward_reward_weight * velocity[0]
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(qp, info)
        reward = forward_reward - ctrl_cost
        state.metrics.update(
            x_position=qp.pos[0, 0],
            x_velocity=velocity[0],
            reward_run=forward_reward,
            reward_ctrl=-ctrl_cost)
        self.state = state.replace(qp=qp, obs=obs, reward=reward) 
        done = self.step_number == 1000
        return self.state.obs, self.state.reward, done, ()

    def get_qps(self):
        return self._qps

env_tasks = {
    "normal":{},
    "carry_stuff":{"torso": 4.,"thigh": 1.,"shin": 1.,"foot": 1.},
    'carry_stuff_hugegravity': {'torso': 4.0,'thigh': 1.0,'shin': 1.0,'foot': 1.0,'gravity': 1.5},
    "defective_module":{"obs_mask":0.5},
    "hugefoot":{"foot":1.5},
    "hugefoot_rainfall": {'foot': 1.5, 'friction': 0.4},
    "inverted_actions":{"action_swap":[0,1,2,3,4,5]},
    "moon":{"gravity":0.15},
    "tinyfoot":{"foot":0.5},
    "tinyfoot_moon": {'foot': 0.5, 'gravity': 0.15},
    "rainfall":{"friction":0.4},
}
# %%
# import jax
# import jax.numpy as jnp

# def generate_rng(seed: int) -> jax.random.PRNGKey:
#     return jax.random.PRNGKey(seed)

# seq_tasks = TASK_SEQS["salina/halfcheetah/forgetting"]
# %%
# env = Halfcheetah()
# rng = jp.random_prngkey(0)
# state = env.reset(rng)
# zero_action = jp.zeros((1, env.action_size))
# next_state = env.step(state, zero_action)

# %%


# for task_idx, dict_task in enumerate(seq_tasks):
#     env = get_single_env(
#             dict_task['task'], 110, randomization=True, 
#             normalize_reward=True
#         )
#     rng = jp.random_prngkey(0)
#     state = env.reset(rng)
#     print(state)
#     zero_action = jp.zeros((env.action_size))
#     next_state = env.step(zero_action)
#     print(next_state)
# %%
