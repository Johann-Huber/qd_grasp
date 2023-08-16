# Created by Giuseppe Paolo 
# Date: 10/12/2020

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_billiard.utils import physics, parameters
from gym_billiard.envs import billiard_env
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# TODO implement logger

import logging

logger = logging.getLogger(__name__)

class Curling(billiard_env.BilliardEnv):
  """
  State is composed of:
  s = ([ball_x, ball_y, joint0_angle, joint1_angle, joint0_speed, joint1_speed])

  The values that these components can take are:
  ball_x, ball_y -> [-1.5, 1.5]
  joint0_angle -> [-pi/2, pi/2]
  joint1_angle -> [-pi, pi]
  joint0_speed, joint1_speed -> [-50, 50]
  """
  def __init__(self, seed=None, max_steps=500):
    super().__init__(seed, max_steps)
    self.goals = np.array([[0.8, .8], [0.8, -0.8]])
    self.goalRadius = [0.4, 0.2]

  def reward_function(self):
    """
    This function calculates the reward based on the final position of the ball.
    Once the ball is in the reward area, the close is to the center, the higher the reward
    :param info:
    :return:
    """
    if self.steps >= self.params.MAX_ENV_STEPS: # If we are at the end of the episode
      ball_pose = self.state[:2]
      for goal_idx, goal in enumerate(self.goals):
        dist = np.linalg.norm(ball_pose - goal)
        if dist <= self.goalRadius[goal_idx]:
          reward = (self.goalRadius[goal_idx] - dist)/self.goalRadius[goal_idx]
          done = True
          self.rew_area = goal_idx
          return reward, done
      return 0, True
    return 0, False