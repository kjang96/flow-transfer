import os
from utils import rollout
import argparse
import joblib
import numpy as np
from matplotlib import pyplot as plt
from flow.core.util import emission_to_csv
import control


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--weight', type=str,
                        help='path to the weight file')
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Number of rollouts we will average over')
    parser.add_argument('--binary', type=str, default="sumo-gui",
                        help='Which sumo binary to use')

    args = parser.parse_args()

    # extract the flow environment
    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']
    func = policy._f_dist

    # if this doesn't work, try the one above it
    unwrapped_env = env._wrapped_env.env.unwrapped

    # Recreate experiment params
    tot_cars = unwrapped_env.vehicles.num_vehicles
    rl_cars = unwrapped_env.vehicles.num_rl_vehicles
    max_path_length = int(env.horizon)
    flat_obs = env._wrapped_env.observation_space.flat_dim
    obs_vars = unwrapped_env.obs_var_labels or []
    num_obs_var = flat_obs / tot_cars

    # Set sumo to make a video
    sumo_params = unwrapped_env.sumo_params
    sumo_params.emission_path = "./test_time_rollout/"
    sumo_params.render = args.binary
    unwrapped_env.restart_sumo(sumo_params=sumo_params)

    # Load data into arrays
    all_obs = np.zeros((args.num_rollouts, max_path_length, flat_obs))
    all_rewards = np.zeros((args.num_rollouts, max_path_length))
    rew = []

    ### changes start
    import ipdb; ipdb.set_trace()
    if args.weight:
        func = args.weight
    controller = control.StraightController(func)

    ### changes end
    for j in range(args.num_rollouts):
        # run a single rollout of the experiment
        path = rollout(env=env, agent=policy, controller=controller)

        # collect the observations and rewards from the rollout
        new_obs = path['observations']
        all_obs[j, :new_obs.shape[0], :new_obs.shape[1]] = new_obs
        new_rewards = path['rewards']
        all_rewards[j, :len(new_rewards)] = new_rewards

        # print the cumulative reward of the most recent rollout
        print("Round {}, return: {}".format(j, sum(new_rewards)))
        rew.append(sum(new_rewards))

    # print the average cumulative reward across rollouts
    print("Average, std return: {}, {}".format(np.mean(rew), np.std(rew)))

    # ensure that a reward_plots folder exists in the directory, and if not,
    # create one
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # create an array of time
    sim_step = unwrapped_env.sumo_params.sim_step
    t = np.arange(max_path_length) * sim_step

    for obs_var_idx in range(int(num_obs_var)):
        if len(obs_vars) < obs_var_idx + 1:
            obs_var = "Observation {0}".format(obs_var_idx)
        else:
            obs_var = obs_vars[obs_var_idx]

        # plot mean value for observation for each vehicle across rollouts
        plt.figure()
        for car in range(tot_cars):
            center = np.mean(all_obs[:, :, tot_cars*obs_var_idx + car], axis=0)
            plt.plot(range(max_path_length), center, lw=2.0,
                     label='Veh {}'.format(car))
        plt.ylabel(obs_var, fontsize=15)
        plt.xlabel("time (s)", fontsize=15)
        plt.title("{2}, Autonomous Penetration: {0}/{1}".
                  format(rl_cars, tot_cars, obs_var), fontsize=16)
        plt.legend(loc=0)

        # save the plot in the "plots" directory
        plt.savefig("plots/{0}_{1}.png".format(args.plotname, obs_var),
                    bbox="tight")

        # plot mean values for the observations across all vehicles and all
        # rollouts
        car_mean = np.mean(np.mean(
            all_obs[:, :, tot_cars*obs_var_idx:tot_cars*(obs_var_idx + 1)],
            axis=0), axis=1)
        plt.figure()
        plt.plot(t, car_mean)
        plt.ylabel(obs_var, fontsize=15)
        plt.xlabel("time (s)", fontsize=15)
        plt.title("Mean {2}, Autonomous Penetration: {0}/{1}".
                  format(rl_cars, tot_cars, obs_var), fontsize=16)

        # save the plot in the "plots" directory
        plt.savefig("plots/{0}_{1}_mean.png".format(args.plotname, obs_var),
                    bbox="tight")

    # Make a figure for the mean rewards over the course of the rollout
    mean_reward = np.mean(all_rewards, axis=0)

    plt.figure()
    plt.plot(t, mean_reward, lw=2.0)
    plt.ylabel("reward", fontsize=15)
    plt.xlabel("time (s)", fontsize=15)
    plt.title("Reward, Autonomous Penetration: {0}/{1}".
              format(rl_cars, tot_cars), fontsize=16)

    # save the rewards plot in the "reward_plots" directory
    plt.savefig("plots/{0}_reward.png".format(args.plotname), bbox="tight")

    # if prompted, convert the emission file into a csv file
    # if args.emission_to_csv:
    #     dir_path = os.path.dirname(os.path.realpath(__file__))
    #     emission_filename = "{0}-emission.xml".format(
    #         unwrapped_env.scenario.name)

    #     emission_path = \
    #         "{0}/test_time_rollout/{1}".format(dir_path, emission_filename)

    #     emission_to_csv(emission_path)
