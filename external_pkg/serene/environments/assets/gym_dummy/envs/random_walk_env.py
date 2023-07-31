# Created by Giuseppe Paolo 
# Date: 16/03/2020

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class RandomWalkEnv(gym.Env):
  """
  Random walk environment. Used to test trajectory evolution algorithms. Each step is decided by the action given
  """
  def __init__(self, seed=None, max_steps=100):
    """ Constructor
    :param seed: the random seed for the environment
    :param max_steps: the maximum number of steps the episode lasts
    :return:
    """
    self.pose = np.zeros(2)
    self.t = 0
    self.max_steps = max_steps
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
    Performs environment step. The action is used to drive the "Random walk"
    :param action:
    :return:
    """
    done = False
    self.pose = np.clip(self.pose + 0.05 * action, -1, 1)
    self.t += 1
    if self.t == self.max_steps:
      done = True

    return self.pose, 0, done, {}

  def reset(self):
    """
    Resets environment
    :return:
    """
    self.pose = np.zeros(2)
    self.t = 0
    return self.pose