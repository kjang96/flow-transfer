Dependencies: Theano, install with: `pip install theano`

This controller takes in a path to a pkl file containing
a theano.compile.function_module.Function object, which has
the weights of this controller embedded in it. Provided with
this class is "function.pkl," a controller designed for an
autonomous vehicle (AV) within the following scenario, which
should be recreated in SmartCity as closely as possible:
    - one straight road
    - one AV behind an IDM vehicle (10 m apart, with noise)

Listed below are the experiment parameters used in this experiment: 
    - target_velocity: 10 m/s
    - speed_limit: 30 m/s
    - max_acceleration: 3 m/s^2
    - max_deacceleration: 3 m/s^2
    - road_length: 2000 m

Observations for this specific experiment should be provided to 
the StraightController function `get_action` in the normalized form: 

[[RL velocity / speed_limit, RL absolute position / road_length],
[IDM velocity / speed_limit, IDM absolute position / road_length]]

Inputs in this observation array must be scaled according to the
experiment parameters listed above for the neural net weights to 
provide accurate accelerations!


---------------------------------------------------------
Example Usage:

sc = StraightController("function.pkl")
observation = [[0.609285, 0.85583362],
               [0.14016343, 0.22482264]]
sc.get_action(observation)