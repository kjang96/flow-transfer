#!/usr/bin/env python2

import joblib

import pickle
import sys

"""
Used for testing
"""

if len(sys.argv) != 2:
    print("Need to supply exactly 1 argument.")
    sys.exit()

file = sys.argv[1]
# file = open(file, 'rb')
# data = pickle.load(file)
# file.close()
data = joblib.load(file)
# import ipdb; ipdb.set_trace()
print(data)
print("Success")
a = 3 


# Traceback (most recent call last):
#   File "Followerstraight.py", line 163, in <module>
#     main()
#   File "Followerstraight.py", line 154, in main
#     control=SC("weight_0.pkl")
#   File "/home/themainframe/catkin_ws/src/line_following/script/control.py", line 29, in __init__
#     self.func = joblib.load(pkl)

