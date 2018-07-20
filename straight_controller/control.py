import pickle
import argparse
import joblib
import numpy as np


class StraightController: 

    """

    This controller takes in a path to a pkl file containing
    a theano.compile.function_module.Function object, which has
    the weights of this controller embedded in it. 

    See README for more specific instructions.

    ---------------------------------------------------------
    Example Usage:

    sc = StraightController("function.pkl")
    observation = [[0.609285, 0.85583362],
                   [0.14016343, 0.22482264]]
    sc.get_action(observation)

    """

    def __init__(self, pkl):
        
        self.func = joblib.load(pkl)


    def get_action(self, observation):
        """
        Maps observations to actions.

        Parameters
        ----------
        observation: 2d list
            - for this specific case, observations should be provided
              in the form of
              [[RL velocity / speed_limit (30), RL absolute position / road_length (2000)],
               [IDM velocity / speed_limit (30), IDM absolute position / road_length (2000)]]

        Returns
        -------
        action: numpy ndarray
            acceleration or deacceleration for agent to take
        """
        obs = np.asarray(observation).T
        flat_obs = obs.flatten()
        mean, log_std = [x[0] for x in self.func([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action


