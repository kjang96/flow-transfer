#!/usr/bin/env python

# import joblib
import pickle
import os
import sys
import dill

"""
FOR USE ON BERKELEY END

Takes a pkl file and re pickles it in a protocol useable by python2.7

In order to test this works, use a conda env customized for python2.7
"""

if len(sys.argv) != 2:
    print("Exactly 1 arg should be supplied")
    sys.exit()

file = sys.argv[1]
output = sys.argv[1]

# checkpoint_file = 'checkpoint-150' # hardcoded af
# checkpoint_path = os.path.join(path, output) 
p = pickle.load(open(file, 'rb'))
new_assembly = pickle.loads(p['evaluator'])
p['evaluator'] = pickle.dumps(new_assembly, protocol=2)
# output = open(new)
pickle.dump(p, open(file, 'wb'), protocol=2)
print('Downgrade success')



