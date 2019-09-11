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

file_name = sys.argv[1]
output = sys.argv[1]

# checkpoint_file = 'checkpoint-150' # hardcoded af
# checkpoint_path = os.path.join(path, output) 
p = pickle.load(open(file_name, 'rb'))
# import ipdb; ipdb.set_trace()
# new_assembly = pickle.loads(p['evaluator'])
new_assembly = pickle.loads(p['worker'])
# p['evaluator'] = pickle.dumps(new_assembly, protocol=2)
p['worker'] = pickle.dumps(new_assembly, protocol=2)
# output = open(new)
pickle.dump(p, open(file_name, 'wb'), protocol=2)


### Now do the same thing with checkpoint.tune_metadata
file_name = file_name + '.tune_metadata'
metadata = pickle.load(open(file_name, 'rb'))
pickle.dump(metadata, open(file_name, 'wb'), protocol=2)

print('Downgrade success')



