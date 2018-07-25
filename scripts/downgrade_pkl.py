#!/usr/bin/env python

import joblib
import pickle
import os
import sys

"""
FOR USE ON BERKELEY END

Takes a pkl file and re pickles it in a protocol useable by python2.7
"""

if len(sys.argv) != 3:
    print("Exactly 2 args should be supplied")
    sys.exit()

file = sys.argv[1]
output = sys.argv[2]

data = joblib.load(file) # error occurs when running this script with python2.7
# policy = data['policy']
# func = policy._f_dist
output = open(output, 'wb')

# THIS STEP SHOULD BE DONE USING PYTHON2.7
# pkl = joblib.dump(data, output, compress=3, protocol=2)
print("Dumping to... ", output)
pickle.dump(data, output, protocol=1)


output.close()
# import ipdb; ipdb.set_trace()
# joblib.dump(func, dest, compress=3)



