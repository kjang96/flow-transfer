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
from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl

from gym.spaces.box import Box


import json
import numpy as np

import ray
import ray.rllib.agents.ppo as ppo
# from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray import tune
from ray.tune.registry import register_env
from ray.tune import run_experiments

# from flow.controllers import ContinuousRouter
from flow.controllers import IDMController
from flow.controllers import RLController
# from flow.controllers import SumoLaneChangeController, ContinuousRouter
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.params import InFlows
from flow.envs.udssc_env import UDSSCMergeEnvReset
# from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
# from flow.core.vehicles import Vehicles
# from flow.scenarios.figure8.figure8_scenario import ADDITIONAL_NET_PARAMS

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder


from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.base_env import BaseEnv

class FakeEnv(BaseEnv):
    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        return np.zeros(94) #{'test': np.zeros(94)}

    def step(self, action_dict):
        # return {'test': np.zeros(94)}, {'test': 1}, {'test': False}, {}
        return np.zeros(94)

    def get_state(self):
        return np.zeros(94)

    @property
    def observation_space(self):
        box = Box(low=0.,
                  high=1,
                  shape=(94,),
                  dtype=np.float32)
        return box

    @property
    def action_space(self):
        return Box(low=-1,
                   high=1,
                   shape=(2,),
                   dtype=np.float32)

def create_env(_):
    return FakeEnv()

class RllibController:
    """
    Example usage: 
    controller = RllibController('../no_noise', 100)

    This is slightly different than the rllab controller in control.py

    Instead of passing a path to a pkl file, pass the path of the directory
    containing checkpoints, as well as the number of the checkpoint you
    intend to reenact. Data meant for this controller will be stored in 
    'data/rllib'.

    For our purposes, checkpoint_num will always be 100, and algo will always
    be 'PPO' 

    If calling as __main__, the run command is:
    python rllib_control.py data/rllib/[some_policy] [100]

    """
    
    def __init__(self, result_dir, checkpoint_num=150, algo='PPO'):

        checkpoint_num = str(checkpoint_num)
        # # config = get_rllib_config(result_dir)
        # # pkl = get_rllib_pkl(result_dir)

        #create_env, env_name = make_create_env(params=flow_params, version=0)

        # Register as rllib env
        register_env('test', create_env)

                                
        obs_space = Box(low=0.,
                    high=1,
                    shape=(94,),
                    dtype=np.float32)

        act_space = Box(low=-1,
                    high=1,
                    shape=(2,),
                    dtype=np.float32)


        # def gen_policy_agent():
        #     return (PPOPolicyGraph, obs_space, act_space, {})

        # def gen_policy_adversary():
        #     return (PPOPolicyGraph, obs_space, adv_action_space, {})

        # <-- old
        # Setup PG with an ensemble of `num_policies` different policy graphs
        # policy_graphs = {'av': gen_policy_agent(), 'adversary': gen_policy_adversary()}

        def policy_mapping_fn(agent_id):
            return agent_id

        # policy_ids = list(policy_graphs.keys())

        config = ppo.DEFAULT_CONFIG.copy()
        config['model'].update({'fcnet_hiddens': [100, 50, 25]})
        config["observation_filter"] = "NoFilter"
        config['simple_optimizer'] = True

        # config.update({
        #     'multiagent': {
        #         'policy_graphs': policy_graphs,
        #         'policy_mapping_fn': tune.function(policy_mapping_fn)
        #     }
        # }) 


        # check if we have a multiagent scenario but in a
        # backwards compatible way
        # if config.get('multiagent', {}).get('policy_graphs', {}):
        #     multiagent = True
        #     config['multiagent'] = pkl['multiagent']
        # else:
        #     multiagent = False

        # Run on only one cpu for rendering purposes
        config['num_workers'] = 0

        # flow_params = get_flow_params(config)

        # # Create and register a gym+rllib env
        # create_env, env_name = make_create_env(
        #     params=flow_params, version=0, render=False)
        # register_env(env_name, create_env)

        # Determine agent and checkpoint
        agent_cls = get_agent_class(algo)


        # create the agent that will be used to compute the actions
        self.agent = agent_cls(env='test', config=config)
        # agent = agent_cls(config=config)
        checkpoint = result_dir + '/checkpoint_' + checkpoint_num
        checkpoint = checkpoint + '/checkpoint-' + checkpoint_num
        self.agent.restore(checkpoint)


        # multiagent = True
        # if multiagent:
        #     rets = {}
        #     # map the agent id to its policy
        #     self.policy_map_fn = config['multiagent']['policy_mapping_fn'].func
        #     for key in config['multiagent']['policy_graphs'].keys():
        #         rets[key] = []
        # else:
        #     rets = []
        #     env_params=env_params, sumo_params=sumo_params, scenario=scenario))

    def get_action(self, state, agent_id='av'):
        """
        Maps observations to actions.

        Parameters
        ----------
        observation: 1d list

        Returns
        -------
        action: numpy ndarray
            acceleration or deacceleration for agent to take
        """
        actions = self.agent.compute_action(state, policy_id=self.policy_map_fn(agent_id))
        actions = np.clip(actions, [-1]*len(actions), [1]*len(actions))
        return actions
    


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1, object_store_memory=10000000)
    c = RllibController(args.result_dir, args.checkpoint_num, algo='PPO')


