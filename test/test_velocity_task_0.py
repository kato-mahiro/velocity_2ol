import sys
import unittest

sys.path.append("../")
from  velocity_task import velocity_env

class Test_velocity_env_0(unittest.TestCase):
    def setUp(self):
        self.myenv = velocity_env()

    def test_reset(self):
        self.myenv.reset()

    def test_step(self):
        self.myenv.reset()

        end_flag = False
        while(not end_flag):
            print(self.myenv.step_num)
            obs, err, done = self.myenv.step(net_output=[1.0])
            print(f"target: {self.myenv.target}, target_v: {self.myenv.target_v}, obs: {obs}, err: {err}, done: {done}")
            end_flag = done