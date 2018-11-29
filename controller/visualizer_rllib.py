"""Visualizer for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
import numpy as np
import os
import sys

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl


EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /tmp/ray/result_dir 1

Here the arguments are:
1 - the number of the checkpoint
"""


def visualizer_rllib(args, algo='PPO', checkpoint_num='100'):

    def compute_action(state, agent, policy_map_fn, agent_id='av'):
        if multiagent:
            action = agent.compute_action(state, policy_id=policy_map_fn(agent_id))
        else:
            action = agent.compute_action(state)
        return action

    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    # config = get_rllib_config(result_dir + '/..')
    # pkl = get_rllib_pkl(result_dir + '/..')
    config = get_rllib_config(result_dir)
    pkl = get_rllib_pkl(result_dir)

    # check if we have a multiagent scenario but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policy_graphs', {}):
        multiagent = True
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)

    # hack for old pkl files
    # TODO(ev) remove eventually
    # sumo_params = flow_params['sumo']
    # setattr(sumo_params, 'num_clients', 1)

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(
        params=flow_params, version=0, render=False)
    register_env(env_name, create_env)
# 
    # Determine agent and checkpoint
    agent_cls = get_agent_class(algo)



    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    # agent = agent_cls(config=config)
    checkpoint = result_dir + '/checkpoint_' + checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + checkpoint_num
    import ipdb; ipdb.set_trace()
    agent.restore(checkpoint)


    if multiagent:
        rets = {}
        # map the agent id to its policy
        policy_map_fn = config['multiagent']['policy_mapping_fn'].func
        for key in config['multiagent']['policy_graphs'].keys():
            rets[key] = []
    else:
        rets = []

    import ipdb; ipdb.set_trace()

    env = ModelCatalog.get_preprocessor_as_wrapper(env_class(
        env_params=env_params, sumo_params=sumo_params, scenario=scenario))

    


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--emission_to_csv',
        action='store_true',
        help='Specifies whether to convert the emission file '
             'created by sumo into a csv file')
    parser.add_argument(
        '--no_render',
        action='store_true',
        help='Specifies whether to visualize the results')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--sumo_web3d',
        action='store_true',
        help='Specifies whether sumo web3d will be used to visualize.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1)
    visualizer_rllib(args)
