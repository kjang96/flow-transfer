# Flow x SmartCity Transfer

## For Rllab ('controller/control.py')
Dependencies: Theano, install with: `pip install theano`

For this experiment's purpose, you should only need to interface with `controller/control.py`. This controller takes in a path to a pkl file containing a theano.compile.function_module.Function object, which has the weights of this controller embedded in it. Provided with this class is "function.pkl," a controller designed for an autonomous vehicle (AV) within the following scenario, which should be recreated in SmartCity as closely as possible. A zoomed-picture of the beginning of the scenario "straight_scenario.png" is attached. 
    - one straight road
    - one AV behind an IDM vehicle (10 m apart, with noise)

Observations for specific experiments should be provided to 
the StraightController function `get_action` in the normalized forms
listed according to the version type below: 

Inputs in this observation array must be scaled according to the
experiment parameters listed for the neural net weights to 
provide accurate accelerations!

IMPORTANT: Certain features Theano for Python2.7 is behind.
- `/.../theano/gof/opt.py`
    - class _metadict 
    - class ChangeTracker
- `/.../theano/compile/function_module.py`
    - class Supervisor

To apply necessary changes for Theano (2.7), run `scripts/apply_patch.py`, which will change the above listed classes to inherit from object.


## For RLlib ('controller/rllib_control.py')
Dependencies: Ray, Flow 

Ray installation information: https://ray.readthedocs.io/en/latest/installation.html

Flow installation: https://flow.readthedocs.io/en/latest/flow_setup.html

For interfacing with RLlib-trained policy, view `controller/rllib_control.py`. This controller takes in a path to a directory containing checkpoints of an RLlib-trained policy. 

--- 
Example Usage:

`sc = StraightController("../data/weights/weight_3.pkl")
observation = [0.00207403, 0., 0.]
sc.get_action(observation)`

---
## Log of Experiment Parameters
Listed below are the experiment parameters used in this experiment: 

### weight_0.pkl (v0)
- target_velocity: 10 m/s
- speed_limit: 30 m/s
- max_acceleration: 3 m/s^2
- max_deacceleration: 3 m/s^2
- road_length: 2000 m
- Observations provided as: 
```[[RL velocity / speed_limit, RL absolute position / road_length],
   [IDM velocity / speed_limit, IDM absolute position / road_length]]
```

### weight_1.pkl, weight_2.pkl, weight_3.pkl (v1) 
- target_velocity: 10 m/s
- speed_limit: 15 m/s
- max_acceleration: 3 m/s^2
- max_deacceleration: 3 m/s^2
- road_length: 1500 m
- Observations provided as: 
```
[RL headway / road_length, RL velocity / speed_limit, IDM velocity / speed_limit]
```

### weight_4.pkl (v2)
- target_velocity: 10 m/s
- speed_limit: 15 m/s
- max_acceleration: 5 m/s^2
- max_deacceleration: -5 m/s^2
- road_length: 1500 m
This is trained on a different reward function with speed mode aggressive. Running 50 rollouts of this policy yields: 
- Average RL action: -0.00 m/s^2
- Average RL headway: 9.88m
- Average RL velocity: 8.76 m/s

### roundabout_46_2018_10_11_04_14_29_0002.pkl (paper)
- Observations provided as a 1D array in the following form. You'll want to reference the Flow Specification for specifics. The expected length is 92.
	```
	state = np.array(np.concatenate([rl_info, rl_info_2,
                                        merge_dists_0, merge_0_vel,
                                        merge_dists_1, merge_1_vel,
                                        queue_0, queue_1,
                                        roundabout_full]))
    ```
    
### roundabout_60_2018_10_12_21_52_40_0004.pkl (paper)
- CHANGES: Some normalizers are different. I increased the scenario length, so:
    - merge_0_norm: 64.32
    - merge_1_norm: 76.55000000000001
    - queue_0_norm: 14
    - queue_1_norm: 17
    - scenario_length: 402.7499999999999
- Observations provided as a 1D array in the following form. You'll want to reference the Flow Specification for specifics. The expected length is 92.
    ```
    state = np.array(np.concatenate([rl_info, rl_info_2,
                                        merge_dists_0, merge_0_vel,
                                        merge_dists_1, merge_1_vel,
                                        queue_0, queue_1,
                                        roundabout_full]))
    ```

### roundabout_70_2018_10_15_00_43_43_0001.pkl (paper)
- CHANGES: Some normalizers are different. I increased the scenario length again, so:
- Observations provided as a 1D array in the following form. You'll want to reference the Flow Specification for specifics. The expected length is 92.
    - merge_0_norm: 74.32
    - merge_1_norm: 86.57
    - queue_0_norm: 16
    - queue_1_norm: 19
    - scenario_length: 442.71
    ```
    state = np.array(np.concatenate([rl_info, rl_info_2,
                                        merge_dists_0, merge_0_vel,
                                        merge_dists_1, merge_1_vel,
                                        queue_0, queue_1,
                                        roundabout_full]))
    ```

### roundabout_78_2018_10_15_10_06_29_0004.pkl (paper)
- CHANGES: Found an error that had 2 wrong in the ALL_EDGES variable which led to a slightly different scenario_lengh
    - scenario_length: 443.2499999999999
- Observations provided as a 1D array in the following form. You'll want to reference the Flow Specification for specifics. The expected length is 92.
    ```
    state = np.array(np.concatenate([rl_info, rl_info_2,
                                        merge_dists_0, merge_0_vel,
                                        merge_dists_1, merge_1_vel,
                                        queue_0, queue_1,
                                        roundabout_full]))
    ```
 
### roundabout_79_2018_10_15_16_35_06_0003.pkl (paper)
- This has a lot of noise added to the state space 
- Observations provided as a 1D array in the following form. You'll want to reference the Flow Specification for specifics. The expected length is 92.
    ```
    state = np.array(np.concatenate([rl_info, rl_info_2,
                                        merge_dists_0, merge_0_vel,
                                        merge_dists_1, merge_1_vel,
                                        queue_0, queue_1,
                                        roundabout_full]))
    ```

### roundabout_89_2018_10_17_15_25_04_0003.pkl 
- This has no RL or state spac noise, only IDM noise 
- Observations provided as a 1D array in the following form. You'll want to reference the Flow Specification for specifics. The expected length is 92.
    ```
    state = np.array(np.concatenate([rl_info, rl_info_2,
                                        merge_dists_0, merge_0_vel,
                                        merge_dists_1, merge_1_vel,
                                        queue_0, queue_1,
                                        roundabout_full]))
    ```


### ecc_13_2018_10_23_06_13_18_0001
- This is a half stochastic policy, with state space noise.
- IDM vehicles enter the Northern inflow with a probability of 50/3600 per timestep.
- IDM vehicles enter the Western inflow with a probability 300/3600 per timestep.
- RL vehicles enter the Northern inflow at a rate of 50 vehicles per hour.
- RL vehicles enter the Western inflow at a rate of 50 vehicles per hour.
- Max speed is altered to 8 m/s
- Observations provided as a 1D array in the following form. You'll want to reference the Flow Specification for specifics. The expected length is 92.
    ```
    state = np.array(np.concatenate([rl_info, rl_info_2,
                                        merge_dists_0, merge_0_vel,
                                        merge_dists_1, merge_1_vel,
                                        queue_0, queue_1,
                                        roundabout_full]))
    ```

### roundabout_no_noise_0.pkl 
- originally ecc_32_2018_11_16_08_19_22_0004
- Deterministic policy: 1 RL followed by 2 human on Northern inflow; 1 RL followed by 3 human on Western inflow
- Changes to state space: 
    - merge_0_norm: 89.32
    - merge_1_norm: 101.57
    - queue_0_norm: 19
    - queue_1_norm: 22
    - total_scenario_length: 503.22

### roundabout_state_noise_0.pkl
- originally ecc_33_2018_11_16_08_20_43_0006
- Deterministic policy: 1 RL followed by 2 human on Northern inflow; 1 RL followed by 3 human on Western inflow

### roundabout_action_noise_0.pkl
- originally ecc_34_2018_11_16_08_21_55_0006
- Deterministic policy: 1 RL followed by 2 human on Northern inflow; 1 RL followed by 3 human on Western inflow

### roundabout_both_noise_0.pkl
- originally ecc_35_2018_11_16_08_22_42_0005
- Deterministic policy: 1 RL followed by 2 human on Northern inflow; 1 RL followed by 3 human on Western inflow


### varied_inflows_no_noise_0.pkl
- originally ecc_49_2018_11_27_07_42_49_0003
- Deterministic/fixed inflows
- Settings from era 'ecc'
- State space change! Append len_inflow_0, len_inflow_1, to state space. This is documented in the Flow Specification

### varied_inflows_state_noise_0.pkl
- originally ecc_50_2018_11_29_03_44_58_0001
- Deterministic/fixed inflows
- Settings from era 'ecc'
- State space change! Append len_inflow_0, len_inflow_1, to state space. This is documented in the Flow Specification

### varied_inflows_action_noise_0.pkl
- originally ecc_51_2018_11_27_07_46_14_0003
- Deterministic/fixed inflows
- Settings from era 'ecc'
- State space change! Append len_inflow_0, len_inflow_1, to state space. This is documented in the Flow Specification

### varied_inflows_both_noise_0.pkl
- originally ecc_52_2018_11_28_23_21_52_0001
- Deterministic/fixed inflows
- Settings from era 'ecc'
- State space change! Append len_inflow_0, len_inflow_1, to state space. This is documented in the Flow Specification





### for flow team to use
python scripts/extract_theano.py data/policies/varied_inflows_both_noise_0.pkl data/weights/varied_inflows_both_noise_0.pkl

python scripts/downgrade_pkl.py data/weights/varied_inflows_both_noise_0.pkl data/weights/varied_inflows_both_noise_0.pkl

