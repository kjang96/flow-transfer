import pickle
import argparse
import joblib
import numpy as np


class StraightController: 

    """
    Dependencies: Theano

    This controller takes in a path to a .pkl file via command-line containing
    a theanoFunction object. 

    The get_action function takes in an observation and returns an action.


    """

    def __init__(self, pkl):
        # import ipdb; ipdb.set_trace()
        # extract the flow environment
        data = joblib.load(pkl)

        # this is gaussianMLPPolicy object. the dependency is required to 
        # unpack this object
        # self.policy = data['policy']

        # this is a theano Function
        self.func = data


    ######
    def get_action(self, observation):
        obs = np.asarray(observation).T
        # flat_obs = self.observation_space.flatten(observation)
        flat_obs = obs.flatten()
        mean, log_std = [x[0] for x in self.func([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')

    args = parser.parse_args()

    c = StraightController(args.file)


    obs = [[0.609285,   0.85583362],
           [0.14016343, 0.22482264]]

    obs = [[0.0,   0.0],
           [0.0, 0.01]]

    # action = controller.get_action(obs)
    action = c.get_action(obs)
    total = 0
    for i in range(30):
        action, _ = c.get_action(obs)
        total += action[0]
        print(action)
    print('Mean is: ', total/30)





if __name__ == "__main__":
    main()



#############################################


# def get_action(self, obs):
    #     """
    #     max_speed = 30.0
    #     total_length = 2000.0

    #     np.asarray(self.get_state()).T

    #     where self.get_state() returns [[rl_vel / max_speed, rl pos / total_length], 
    #                                     [idm_vel / max_speed, idm_pos / total_length]]

    #     """
    #     # sample_obs = np.array([[0, 0], [0, 0.1]]).T
    #     action, agent_info = self.policy.get_action(obs)
    #     return action