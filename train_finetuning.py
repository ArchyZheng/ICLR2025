from ast import Delete
import os
import random
import time

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl.agents import SACLearner
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate_cl
from jaxrl.utils import Logger

from continual_world import TASK_SEQS, get_single_env

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', "cw2-ab-coffee-button", 'Environment name.')
flags.DEFINE_string('save_dir', '/home/yijunyan/Data/PyCode/MORE/src/jaxrl/logs/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 66, 'Random seed.')
flags.DEFINE_string('base_algo', 'comps', 'base learning algorithm')

flags.DEFINE_string('env_type', 'deterministic', 'The type of env is either deterministic or random_init_all')
flags.DEFINE_boolean('normalize_reward', False, 'Normalize rewards')
flags.DEFINE_integer('eval_episodes', 1, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps for each task')
flags.DEFINE_integer('start_training', int(1e4), 'Number of training steps to start training.')

flags.DEFINE_integer('buffer_size', int(1e6), 'Size of replay buffer')

flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_string('wandb_mode', 'online', 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "jaxrl_comps", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', None, "the entity (team) of wandb's project")
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_cw.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    # config tasks
    seq_tasks = TASK_SEQS[FLAGS.env_name]

    kwargs = dict(FLAGS.config)
    algo = FLAGS.base_algo
    run_name = f"{FLAGS.env_name}__{algo}__{FLAGS.seed}__{int(time.time())}"
    save_policy_dir = f"logs/saved_actors/{run_name}.json"

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
    temp_env = get_single_env(
        TASK_SEQS[FLAGS.env_name][0]['task'], FLAGS.seed, 
        randomization=FLAGS.env_type)
    if algo == 'comps':
        agent = SACLearner(
            FLAGS.seed,
            temp_env.observation_space.sample()[np.newaxis],
            temp_env.action_space.sample()[np.newaxis], 
            **kwargs)
        del temp_env
    else:
        raise NotImplementedError()

    # continual learning loop
    eval_envs = []
    for idx, dict_t in enumerate(seq_tasks):
        # only for ablation study
        if idx == 0:
            eval_seed = 60
        else:
            eval_seed = FLAGS.seed
        eval_envs.append(get_single_env(dict_t['task'], eval_seed, randomization=FLAGS.env_type))

    # continual learning loop
    total_env_steps = 0
    for idx, dict_t in enumerate(seq_tasks):
        print(f'Learning on task {idx+1}: {dict_t["task"]} for {FLAGS.max_steps} steps')

        '''
        Learning subroutine for the current task
        '''
        # only for ablation study
        if idx == 0:
            expl_seed = 60
        else:
            expl_seed = FLAGS.seed
        # set continual world environment
        env = get_single_env(
            dict_t['task'], expl_seed, randomization=FLAGS.env_type, 
            normalize_reward=FLAGS.normalize_reward)

        # reset replay buffer
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                    FLAGS.buffer_size or FLAGS.max_steps)

        observation, done = env.reset(), False
        for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                        smoothing=0.1,
                        disable=not FLAGS.tqdm):
            if i < FLAGS.start_training:
                action = env.action_space.sample()
            else:
                action = agent.sample_actions(observation[np.newaxis])
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

            replay_buffer.insert(observation, action, reward, mask, float(done),
                                next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    wandb.log({f'training/{k}': v, 'global_steps': total_env_steps})

            if (i >= FLAGS.start_training) and (i % FLAGS.updates_per_step == 0):
                for _ in range(FLAGS.updates_per_step):
                    batch = replay_buffer.sample(FLAGS.batch_size)
                    update_info = agent.update(batch)

                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        wandb.log({f'training/{k}': v, 'global_steps': total_env_steps})

            if i % FLAGS.eval_interval == 0:
                eval_stats = evaluate_cl(agent, eval_envs, FLAGS.eval_episodes, naive_sac=True)

                for k, v in eval_stats.items():
                    wandb.log({f'evaluation/average_{k}s': v, 'global_steps': total_env_steps})

                # Update the log with collected data
                eval_stats['cl_method'] = algo
                eval_stats['x'] = total_env_steps
                eval_stats['steps_per_task'] = FLAGS.max_steps
                log.update(eval_stats) 
        
        '''
        Updating miscellaneous things
        '''
        print('End the current task')
        agent.end_task(save_policy_dir)

    # save log data
    log.save()

if __name__ == '__main__':
    app.run(main)
