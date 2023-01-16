import sys
import unittest

sys.path.append("../")
from  velocity_task import velocity_env

class Test_velocity_env_0(unittest.TestCase):
    def setUp(self):
        self.myenv = velocity_env(order = 0)
        assert self.myenv.order == 0

    def test_reset(self):
        self.myenv.reset()
        self.assertEqual( self.myenv.target_v, 0.5 )
        self.assertEqual( self.myenv.a, 0 )

    def test_step(self):
        self.myenv.reset()

        end_flag = False
        while(not end_flag):
            print(self.myenv.step_num)
            obs, err, done = self.myenv.step(net_output=[1.0])
            print(f"obs {obs}, err {err}, done {done}")
            end_flag = done

class Test_velocity_env_1(unittest.TestCase):
    def setUp(self):
        self.myenv = velocity_env(order = 1)
        assert self.myenv.order == 1

    def test_reset(self):
        self.myenv.reset()
        assert( 0.0 < self.myenv.target_v and self.myenv.target_v <= 1.0)
        self.assertEqual( self.myenv.a, 0 )

    def test_step(self):
        self.myenv.reset()

        end_flag = False
        while(not end_flag):
            print(self.myenv.step_num)
            obs, err, done = self.myenv.step(net_output=[1.0])
            print(f"obs {obs}, err {err}, done {done}")
            end_flag = done

class Test_velocity_env_2(unittest.TestCase):
    def setUp(self):
        self.myenv = velocity_env(order = 2)
        assert self.myenv.order == 2

    def test_reset(self):
        self.myenv.reset()

    def test_step(self):
        self.myenv.reset()

        end_flag = False
        while(not end_flag):
            assert(self.myenv.a >= -1.0 or self.myenv.a <= 1.0 )
            assert(0.0 <= self.myenv.target_v and self.myenv.target_v <= 1.0)
            print(self.myenv.step_num)
            obs, err, done = self.myenv.step(net_output=[1.0])
            print(f"obs {obs}, err {err}, done {done}, v: {self.myenv.target_v}, a: {self.myenv.a}")
            end_flag = done