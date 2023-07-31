# Created by Giuseppe Paolo 
# Date: 13/03/2020

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class DummyEnv(gym.Env):
  """
  Dummy environment. Used to test evolution algorithms. The observation corresponds to the action.
  """
  def __init__(self, seed=None, max_steps=1):
    """ Constructor
    :param seed: the random seed for the environment
    :param max_steps: the maximum number of steps the episode lasts
    :return:
    """
    self.max_Steps = max_steps
    self.observation_space = spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32)
    self.action_space = spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32)
    self.seed(seed)

  def seed(self, seed=None):
    """
    Function to seed the environment
    :param seed: The random seed
    :return: [seed]
    """
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    """
    Performs environment step
    :param action:
    :return:
    """
    obs = np.clip(action, -1, 1)
    reward = 0
    if np.all(0.8 >= obs) and np.all(obs >= 0.75):
      reward = 1
    return obs, reward, True, {}

  def reset(self):
    """
    Resets environment
    :return:
    """
    return np.zeros(2)