# Created by Giuseppe Paolo 
# Date: 13/03/2020

from gym.envs.registration import register

register(
    id='Dummy-v0',
    entry_point='gym_dummy.envs:DummyEnv',
    # timestep_limit=1000,
)

register(
    id='Walker2D-v0',
    entry_point='gym_dummy.envs:Walker2DEnv',
)