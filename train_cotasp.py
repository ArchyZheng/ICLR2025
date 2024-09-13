'''
CONTINUAL TASK ALLOCATION IN META-POLICY NETWORK VIA SPARSE PROMPTING
'''

import itertools
import random
import time

import numpy as np
import wandb
import yaml
from absl import app, flags
from ml_collections import config_flags, ConfigDict

from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate_cl
from jaxrl.utils import Logger
from jaxrl.agents.sac.sac_learner import CoTASPLearner
from jaxrl.agents.sac.sac_mask_combination import MaskCombinationLearner
from continual_world import TASK_SEQS, get_single_env
import jax.numpy as jnp
from jax.tree_util import tree_map
from flax.core import unfreeze, FrozenDict

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'cw10', 'Environment name.')
flags.DEFINE_integer('seed', 110, 'Random seed.')
flags.DEFINE_string('base_algo', 'cotasp', 'base learning algorithm')

flags.DEFINE_string('env_type', 'random_init_all', 'The type of env is either deterministic or random_init_all')
flags.DEFINE_boolean('normalize_reward', True, 'Normalize rewards')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 200, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updating per # environment steps.')
flags.DEFINE_integer('buffer_size', int(1e6), 'Size of replay buffer')
flags.DEFINE_integer('max_step', int(1e6), 'Number of training steps for each task')
flags.DEFINE_integer('start_training', int(1e4), 'Number of training steps to start training.')
flags.DEFINE_integer('theta_step', int(990), 'Number of training steps for theta.')
flags.DEFINE_integer('alpha_step', int(10), 'Number of finetune steps for alpha.')

flags.DEFINE_boolean('rnd_explore', True, 'random policy distillation')
flags.DEFINE_integer('distill_steps', int(2e4), 'distillation steps')

flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_string('wandb_mode', 'online', 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "check repeatness", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', None, "the entity (team) of wandb's project")
flags.DEFINE_boolean('save_checkpoint', False, 'Save meta-policy network parameters')
flags.DEFINE_string('save_dir', '~/rl-archy/Documents/PyCode/CoTASP/logs', 'Logging dir.')

flags.DEFINE_integer('calculate_layer_sensitivity_interval', int(8e4), 'calculate the layer sensitivity every x steps')
flags.DEFINE_integer('evaluation_batch_size', int(1e3), "the batch size for evaluation")
flags.DEFINE_float('layer_neuron_threshold', 0.4, 'the threshold to reset the parameters')


# YAML file path to cotasp's hyperparameter configuration
with open('configs/sac_cotasp.yaml', 'r') as file:
    yaml_dict = yaml.unsafe_load(file)
config_flags.DEFINE_config_dict(
    'config',
    ConfigDict(yaml_dict),
    'Training hyperparameter configuration.',
    lock_config=False
)

def main(_):
    # config tasks
    seq_tasks = TASK_SEQS[FLAGS.env_name]
    algo_kwargs = dict(FLAGS.config)
    algo = FLAGS.base_algo
    run_name = f"{FLAGS.env_name}__{algo}__{FLAGS.seed}__fixed_lambda:{algo_kwargs['dict_configs_fixed']['alpha']}__random_lambda:{algo_kwargs['dict_configs_random']['alpha']}__{int(time.time())}"

    if FLAGS.save_checkpoint:
        save_policy_dir = f"logs/saved_actors/{run_name}.json"
        save_dict_dir = f"logs/saved_dicts/{run_name}"
    else:
        save_policy_dir = None
        save_dict_dir = None

    wandb.init(
        project=FLAGS.wandb_project_name,
        entity=FLAGS.wandb_entity,
        sync_tensorboard=True,
        config=FLAGS,
        name=run_name,
        monitor_gym=False,
        save_code=False,
        mode=FLAGS.wandb_mode,
        dir=FLAGS.save_dir
    )
    wandb.config.update({"algo": algo})

    log = Logger(wandb.run.dir)

    # random numpy seeding
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # initialize SAC agent
    algo_kwargs['update_coef'] = False
    algo_kwargs['update_dict'] = False
    temp_env = get_single_env(
        TASK_SEQS[FLAGS.env_name][0]['task'], FLAGS.seed, 
        randomization=FLAGS.env_type)
    if algo == 'cotasp':
        # agent = CoTASPLearner(
        agent = MaskCombinationLearner(
            FLAGS.seed,
            temp_env.observation_space.sample()[np.newaxis],
            temp_env.action_space.sample()[np.newaxis], 
            len(seq_tasks),
            **algo_kwargs)
        del temp_env
    else:
        raise NotImplementedError()
    
    '''
    continual learning loop
    '''
    eval_envs = []
    for idx, dict_task in enumerate(seq_tasks):
        eval_envs.append(get_single_env(dict_task['task'], FLAGS.seed, randomization=FLAGS.env_type))

    total_env_steps = 0
    for task_idx, dict_task in enumerate(seq_tasks):
        
        '''
        Learning subroutine for the current task
        '''
        print(f'Learning on task {task_idx+1}: {dict_task["task"]} for {FLAGS.max_step} steps')
        # start the current task
        agent.start_task(task_idx, dict_task["hint"])
        # >>>>>>>>>>>>>>>>>>>> store the parameter before learning the new task >>>>>>>>>>>>>>>
        temp_params = agent.get_params() # NOTE: store the parameter before learning the new task
        # <<<<<<<<<<<<<<<<<<<< store the parameter before learning the new task <<<<<<<<<<<<<<<
        
        if task_idx > 0 and FLAGS.rnd_explore:
            '''
            (Optional) Rand policy distillation for better exploration in the initial stage
            '''
            for i in range(FLAGS.distill_steps):
                batch = replay_buffer.sample(FLAGS.batch_size)
                distill_info = agent.rand_net_distill(task_idx, batch)
                
                if i % (FLAGS.distill_steps // 10) == 0:
                    print(i, distill_info)
            # reset actor's optimizer
            agent.reset_actor_optimizer()

        # set continual world environment
        env = get_single_env(
            dict_task['task'], FLAGS.seed, randomization=FLAGS.env_type, 
            normalize_reward=FLAGS.normalize_reward
        )
        # reset replay buffer
        replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, FLAGS.buffer_size or FLAGS.max_step
        )
        evaluation_buffer = ReplayBuffer(
            env.observation_space, env.action_space, FLAGS.evaluation_batch_size
        )
        # reset scheduler
        schedule = itertools.cycle([False]*FLAGS.theta_step + [True]*FLAGS.alpha_step)
        # reset environment
        observation, done = env.reset(), False
        for idx in range(FLAGS.max_step):
            if idx < FLAGS.start_training:
                # initial exploration strategy proposed in ClonEX-SAC
                if task_idx == 0:
                    action = env.action_space.sample()
                else:
                    # uniform-previous strategy
                    mask_id = np.random.choice(task_idx)
                    action = agent.sample_actions(observation[np.newaxis], mask_id)
                    action = np.asarray(action, dtype=np.float32).flatten()
                
                # default initial exploration strategy
                # action = env.action_space.sample()
            else:
                action = agent.sample_actions(observation[np.newaxis], task_idx)
                action = np.asarray(action, dtype=np.float32).flatten()
                
            next_observation, reward, done, info = env.step(action)
            # counting total environment step
            total_env_steps += 1

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
            # only for meta-world
            assert mask == 1.0

            replay_buffer.insert(
                observation, action, reward, mask, float(done), next_observation
            )
            evaluation_buffer.insert(
                observation, action, reward, mask, float(done), next_observation
            )
            # CRUCIAL step easy to overlook
            observation = next_observation

            if done:
                # EPISODIC ending
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    wandb.log({f'training/{k}': v, 'global_steps': total_env_steps})

            if (idx >= FLAGS.start_training) and (idx % FLAGS.updates_per_step == 0):
                for _ in range(FLAGS.updates_per_step):
                    batch = replay_buffer.sample(FLAGS.batch_size)
                    update_info = agent.update(task_idx, batch, next(schedule))
                if idx % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        wandb.log({f'training/{k}': v, 'global_steps': total_env_steps})

            if idx % FLAGS.eval_interval == 0:
                eval_stats = evaluate_cl(agent, eval_envs, FLAGS.eval_episodes)

                for k, v in eval_stats.items():
                    wandb.log({f'evaluation/{k}': v, 'global_steps': total_env_steps})

                # Update the log with collected data
                eval_stats['cl_method'] = algo
                eval_stats['x'] = total_env_steps
                eval_stats['steps_per_task'] = FLAGS.max_step
                log.update(eval_stats)
            # >>>>>>>>>>>>>>>>>>>> calculate the layer sensitivity <<<<<<<<<<<<<<<
            if idx % FLAGS.calculate_layer_sensitivity_interval == 0 and idx > 0:
                batch = evaluation_buffer.sample(FLAGS.evaluation_batch_size)
                temp_observations = jnp.array(batch.observations)
                # dormant, info = calculate_layer_neuron_dormant(temp_observations, agent.actor, task_idx)
                # layer_neuron_difference = get_each_layer_neuron_difference(dormant, info)
                delta_y, delta_mean, info = calculate_layer_neuron(temp_observations, agent.actor, task_idx)
                layer_neuron_difference = get_each_layer_neuron_difference(delta_y,info)
                each_layer_reset_indices = get_each_layer_reset_indices(layer_neuron_difference, threshold=FLAGS.layer_neuron_threshold) # this function will indicate the indices of neurons to reset of each layer
                total_reset_neurons = 0
                total_neurons = 0
                for layer_name, indices in each_layer_reset_indices.items():
                    layer_total_neurons = info['masks'][layer_name][0].sum()
                    reset_percentage = (len(indices) / layer_total_neurons) * 100
                    total_reset_neurons += len(indices)
                    total_neurons += layer_total_neurons
                    wandb.log({
                        f"neuron_reset_indices/{layer_name}": len(indices),
                        f"neuron_reset_percentage/{layer_name}": reset_percentage,
                        'global_steps': total_env_steps
                    })
                
                total_reset_percentage = (total_reset_neurons / total_neurons) * 100
                wandb.log({
                    "total_neuron_reset_indices": total_reset_neurons,
                    "total_neuron_reset_percentage": total_reset_percentage,
                    'global_steps': total_env_steps
                })
                available_indices = get_available_indices(info)
                reset_params = reset_params_four_layers(agent.actor, temp_params, each_layer_reset_indices, available_indices)
                agent.actor = agent.actor.replace(params=reset_params)
            # <<<<<<<<<<<<<<<<<<<< calculate the layer sensitivity <<<<<<<<<<<<<<<
    
        '''
        Updating miscellaneous things
        '''
        print('End of the current task')
        dict_stats = agent.end_task(task_idx, save_policy_dir, save_dict_dir)

    # save log data
    log.save()
    np.save(f'{wandb.run.dir}/dict_stats.npy', dict_stats)

if __name__ == '__main__':
    app.run(main)

def calculate_layer_neuron(observations, actor, task_id):
    info, intermediate = actor(observations, jnp.array([task_id]))
    # noise = jax.random.normal(jax.random.PRNGKey(0), shape=(observations.shape[0], 12))
    noise = observations.mean(axis=0) * 0.01
    # Add the noise to the observations
    perturbed_observations = observations.at[:, :12].add(noise)
    info_, perturbed_intermediate = actor(perturbed_observations, jnp.array([task_id]))
    delta_y = tree_map(lambda x, y: jnp.abs(x - y), intermediate['intermediates'], perturbed_intermediate['intermediates'])
    return delta_y, jnp.abs(info[1]['means'] - info_[1]['means']), info[1]

def get_each_layer_neuron_difference(delta_y, info):
    layer_name = ['backbones_0', 'backbones_1', 'backbones_2', 'backbones_3']
    output = {}
    for layer_name in layer_name:
        denominator = tree_map(lambda x, y: x * y, delta_y[layer_name]['__call__'][0], info['masks'][layer_name])
        denominator =  denominator.mean(axis=0).sum() / info['masks'][layer_name][0].sum()
        molecule = delta_y[layer_name]['__call__'][0].mean(axis=0)
        molecule = jnp.where(info['masks'][layer_name][0] == 1, molecule, 0)
        # Drop the values which are 0 from molecule
        molecule = molecule[molecule != 0]
        output[layer_name] = molecule / denominator
    return output

def get_each_layer_reset_indices(neuron_difference, threshold=0.01):
    output = {}
    for layer_name in ['backbones_0', 'backbones_1', 'backbones_2', 'backbones_3']:
        temp = []
        for i in range(neuron_difference[layer_name].shape[0]):
            if neuron_difference[layer_name][i] < threshold:
                temp.append(i)
        output[layer_name] = jnp.array(temp)
    return output

def get_available_indices(info):
    output = {}
    for layer_name in ['backbones_0', 'backbones_1', 'backbones_2', 'backbones_3']:
        flag = info['masks'][layer_name][0]
        available_indices = jnp.where(flag == 1)[0]
        output[layer_name] = available_indices
    return output

def get_new_params_by_layer(actor, temp_params, reset_indices, layer_name, available_indices):
    reset_indices = available_indices[layer_name][reset_indices[layer_name]]
    flag = jnp.zeros(actor.params[layer_name]['kernel'].T.shape)
    flag = flag.at[reset_indices].set(1)
    params = tree_map(lambda x, y, flag: (1 - flag) * x + flag * y, actor.params[layer_name]['kernel'].T, temp_params[layer_name]['kernel'].T, flag)
    params = params.T
    return params

def reset_params_four_layers(actor, temp_params, reset_indices, available_indices):
    temp_params = unfreeze(temp_params)
    layer_name = ['backbones_0', 'backbones_1', 'backbones_2', 'backbones_3']
    for layer_name in layer_name:
        if reset_indices[layer_name].size > 0:
            new_layer_params = get_new_params_by_layer(actor, temp_params, reset_indices, layer_name, available_indices)
            temp_params[layer_name]['kernel'] = new_layer_params
    return FrozenDict(temp_params)