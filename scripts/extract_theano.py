import joblib
import argparse
import os
import sys

"""
FOR USE ON BERKELEY END

Takes in a path to pkl file embedding a [GaussianMLPPolicy],
creates a pkl file embedding a [TheanoFunction]

Defaults to placing weights in the ../weights/ directory
"""

if len(sys.argv) != 3:
    # import ipdb; ipdb.set_trace()
    print("Exactly 2 args should be supplied")
    sys.exit()

origin = sys.argv[1]
dest = sys.argv[2]
dest = os.path.join("../data/weights/", dest)

data = joblib.load(origin)
policy = data['policy']
func = policy._f_dist

joblib.dump(func, dest, compress=3)



