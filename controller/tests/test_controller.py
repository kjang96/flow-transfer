import numpy as np
import unittest
import sys
import os

from controller.control import StraightController

class TestStraightController(unittest.TestCase):
    def setUp(self):
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, "weight_2.pkl")
        self.sc = StraightController(path)

    def tearDown(self):
        pass

    def test_it_works(self):
        obs = np.array([0., 0., 0.])
        action = self.sc.get_action(obs)
        # print(action[0])
        self.assertIsNotNone(action[0])

if __name__ == "__main__":
    unittest.main()
