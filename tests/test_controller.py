import numpy as np
import unittest
from delaware.controller.control import StraightController

class TestStraightController(unittest.TestCase):
    def setUp(self):
        self.sc = StraightController("../data/weights/weight_2.pkl")

    def tearDown(self):
        pass

    def test_it_works(self):
        obs = np.array([0., 0., 0.])
        action = self.sc.get_action(obs)
        # print(action[0])
        self.assertIsNotNone(action[0])

if __name__ == "__main__":
    unittest.main()
