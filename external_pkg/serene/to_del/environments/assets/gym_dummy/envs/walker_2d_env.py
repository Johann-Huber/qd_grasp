# Created by Giuseppe Paolo 
# Date: 16/03/2020

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from skimage import draw

class Walker2DEnv(gym.Env):
  """
  Random walk environment. Used to test trajectory evolution algorithms. Each step is decided by the action given
  """
  def __init__(self, seed=None, max_steps=50):
    """ Constructor
    :param seed: the random seed for the environment
    :param max_steps: the maximum number of steps the episode lasts
    :return:
    """
    self.pose = np.zeros(2)
    self.t = 0
    self.max_steps = max_steps
    self.observation_space = spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float64)
    self.action_space = spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float64)
    self.seed(seed)
    self.resolution = [64, 64]
    self.center = np.array(self.resolution) / 2

    self.goals = [[(0.75, 0.75), (0.8, 0.8)],
                  # [(-0.11, -0.11), (-0.06, -0.06)],
                  # [(0.3, 0.2), (0.35, 0.25)],
                  [(-.98, .9), (-.93, .95)],
                  # [(-.9, -.9), (-.85, -.85)],
                  # [(.5, -.6), (.55, -.55)],
                  [(0, -.75), (.05, -.7)],
                  # [(-.5, .7), (-.45, .75)],
                  [(-.75, -0.05), (-.7, .0)]
                  ]
    self.max_rew_area = 0.1 * 0.1
    self.reward_area = None

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
    reward = self.calculate_reward()

    if self.t == self.max_steps:
      done = True

    return self.pose, reward, done, {'rew_area': self.reward_area}

  def calculate_reward(self):
    """
    This function calculates the reward
    :return:
    """
    for idx, goal in enumerate(self.goals):
      if goal[1][0] >= self.pose[0] >= goal[0][0] and goal[1][1] >= self.pose[1] >= goal[0][1]:
        # Only give reward if in the same reward area
        if self.reward_area is None or idx == self.reward_area:
          self.reward_area = idx
          return 1
    return 0

  def reset(self):
    """
    Resets environment
    :return:
    """
    self.pose = np.zeros(2)
    self.t = 0
    self.reward_area = None
    return self.pose

  def render(self, mode='human'):
    """
    Class that renders the position of the walker on the plane
    :param mode:
    :return:
    """
    if mode=='human':
      return None
    else:
      array = np.zeros(self.resolution + [3])
      for goal in self.goals:
        pixel_coord = (goal * self.center) + self.center
        # print("Goal: {} - Coord: {}".format(goal, pixel_coord))
        rr, cc = draw.rectangle(pixel_coord[0], pixel_coord[1], shape=array.shape)
        array[rr.astype(int), cc.astype(int)] = np.array([1, 0, 0])

      pixel_coord = (self.pose * self.center) + self.center
      rr, cc = draw.circle(pixel_coord[0], pixel_coord[1], radius=2, shape=array.shape)
      array[rr, cc] = np.array([0, 0, 1]) # Paint circle in red
      return array


