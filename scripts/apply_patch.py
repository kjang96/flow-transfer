import theano
import os
import sys
import subprocess

# import ipdb; ipdb.set_trace()
if sys.version_info[0] >= 3:
	print("No patch necessary for Python 3")
	sys.exit()


"""
a) /usr/local/lib/python2.7/site-packages/theano/__init__.pyc
b) /usr/local/lib/python2.7/site-packages/theano

To be called from the same directory as .patch files

"""

path = os.path.split(theano.__file__)[0] # leads to b)

patch_path = os.getcwd() # should be something like '/Users/kathyjang/research/delaware/temp'


# Apply opt.patch
opt_path = os.path.join(path, "gof", "opt.py")
command = ["patch", opt_path, os.path.join(patch_path, "opt.patch")]
subprocess.Popen(command)

# Apply function_module.patch
func_path = os.path.join(path, "compile", "function_module.py")
command = ["patch", func_path, os.path.join(patch_path, "function_module.patch")]
subprocess.Popen(command)
