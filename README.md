Dependencies: Theano, install with: `pip install theano`

This controller takes in a path to a pkl file containing
a theano.compile.function_module.Function object, which has
the weights of this controller embedded in it. Provided with
this class is "function.pkl," a controller designed for an
autonomous vehicle (AV) within the following scenario, which
should be recreated in SmartCity as closely as possible. A zoomed-
picture of the beginning of the scenario, "straight_scenario.png"
is attached. 
    - one straight road
    - one AV behind an IDM vehicle (10 m apart, with noise)

Observations for specific experiments should be provided to 
the StraightController function `get_action` in the normalized forms
listed according to the version type below: 

Inputs in this observation array must be scaled according to the
experiment parameters listed for the neural net weights to 
provide accurate accelerations!


--- 
Example Usage:

sc = StraightController("../data/weights/weight_0.pkl")
observation = [0.00207403, 0., 0.]
sc.get_action(observation)

---
## Log of Experiment Parameters
Listed below are the experiment parameters used in this experiment: 

function.pkl (v0)
- target_velocity: 10 m/s
- speed_limit: 30 m/s
- max_acceleration: 3 m/s^2
- max_deacceleration: 3 m/s^2
- road_length: 2000 m
- Observations provided as: 
```[[RL velocity / speed_limit, RL absolute position / road_length],
   [IDM velocity / speed_limit, IDM absolute position / road_length]]
```

weight_0.pkl (v1)
- target_velocity: 10 m/s
- speed_limit: 15 m/s
- max_acceleration: 3 m/s^2
- max_deacceleration: 3 m/s^2
- road_length: 1500 m
- Observations provided as: 
```
[RL headway / road_length, RL velocity / speed_limit, IDM velocity / speed_limit]
```

