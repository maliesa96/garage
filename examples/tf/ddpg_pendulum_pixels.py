#!/usr/bin/env python3
"""DDPG example with pixel observations using InvertedDoublePendulum-v2."""

import click
import gym
import tensorflow as tf

from garage.envs.wrappers import Grayscale
from garage.envs.wrappers import MaxAndSkip
from garage.envs.wrappers import PixelObservation
from garage.envs.wrappers import Resize
from garage.envs.wrappers import StackFrames
from garage.experiment import run_experiment
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousCNNPolicy
from garage.tf.q_functions import ContinuousCNNQFunction


def run_task(snapshot_config, variant_data, *_):
    """Run task.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
        variant_data (dict): Custom arguments for the task.
        *_ (object): Ignored by this function.

    """
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:

        env = PixelObservation(gym.make('InvertedDoublePendulum-v2'))
        env = Grayscale(env)
        env = Resize(env, 86, 86)
        env = MaxAndSkip(env, skip=4)
        env = StackFrames(env, 2)
        env = TfEnv(env)

        action_noise = OUStrategy(env.spec, sigma=0.2)

        policy = ContinuousCNNPolicy(env_spec=env.spec,
                                     filter_dims=(3, 7, 5),
                                     num_filters=(32, 64, 64),
                                     strides=(4, 2, 1),
                                     hidden_sizes=[64, 64],
                                     hidden_nonlinearity=tf.nn.relu)
        qf = ContinuousCNNQFunction(env_spec=env.spec,
                                    filter_dims=(3, 7, 5),
                                    num_filters=(32, 64, 64),
                                    strides=(4, 2, 1),
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=tf.nn.relu)

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=variant_data['buffer_size'],
            time_horizon=1)

        ddpg = DDPG(env_spec=env.spec,
                    policy=policy,
                    policy_lr=1e-4,
                    qf_lr=1e-3,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    steps_per_epoch=20,
                    target_update_tau=1e-2,
                    n_train_steps=50,
                    discount=0.9,
                    min_buffer_size=int(1e4),
                    exploration_strategy=action_noise,
                    policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                    qf_optimizer=tf.compat.v1.train.AdamOptimizer,
                    flatten_obses=False)

        runner.setup(algo=ddpg, env=env)

        runner.train(n_epochs=500, batch_size=100)


@click.command()
@click.option('--buffer_size', type=int, default=int(5e4))
def _args(buffer_size):
    """A click command to parse arguments for automated testing purposes.

    Args:
        buffer_size (int): Size of replay buffer.

    Returns:
        int: The input argument as-is.

    """
    return buffer_size


replay_buffer_size = _args.main(standalone_mode=False)
run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
    plot=True,
    variant={'buffer_size': replay_buffer_size},
)
