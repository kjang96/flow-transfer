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


class RllibController:
    
    def __init__(self, result_dir, checkpoint_num='100', algo='PPO'):

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

        # Create and register a gym+rllib env
        create_env, env_name = make_create_env(
            params=flow_params, version=0, render=False)
        register_env(env_name, create_env)
    # 
        # Determine agent and checkpoint
        agent_cls = get_agent_class(algo)



        # create the agent that will be used to compute the actions
        self.agent = agent_cls(env=env_name, config=config)
        # agent = agent_cls(config=config)
        checkpoint = result_dir + '/checkpoint_' + checkpoint_num
        checkpoint = checkpoint + '/checkpoint-' + checkpoint_num
        self.agent.restore(checkpoint)


        if multiagent:
            rets = {}
            # map the agent id to its policy
            self.policy_map_fn = config['multiagent']['policy_mapping_fn'].func
            for key in config['multiagent']['policy_graphs'].keys():
                rets[key] = []
        else:
            rets = []
        #     env_params=env_params, sumo_params=sumo_params, scenario=scenario))

    def get_action(self, state, agent_id='av'):
        action = self.agent.compute_action(state, policy_id=self.policy_map_fn(agent_id))
        # if multiagent:
        #     action = self.agent.compute_action(state, policy_id=self.policy_map_fn(agent_id))
        # else:
        #     action = self.agent.compute_action(state)
        return action
    


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

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1)
    c = RllibController(args.result_dir, args.checkpoint_num, algo='PPO')
    

