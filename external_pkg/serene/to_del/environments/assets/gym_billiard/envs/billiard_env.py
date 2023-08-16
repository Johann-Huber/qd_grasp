# Created by Giuseppe Paolo 
# Date: 21/12/18

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_billiard.utils import physics, parameters
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# TODO implement logger

import logging

logger = logging.getLogger(__name__)


class BilliardEnv(gym.Env):
  """
  State is composed of:
  s = ([ball_x, ball_y], [joint0_angle, joint1_angle], [joint0_speed, joint1_speed])

  The values that these components can take are:
  ball_x, ball_y -> [-1.5, 1.5]
  joint0_angle -> [-pi/2, pi/2]
  joint1_angle -> [-pi, pi]
  joint0_speed, joint1_speed -> [-50, 50]
  """
  metadata = {'render.modes': ['human'],
              'video.frames_per_second': 15
              }

  def __init__(self, seed=None, max_steps=500):
    """ Constructor
    :param seed: the random seed for the environment
    :param max_steps: the maximum number of steps the episode lasts
    :return:
    """
    self.screen = None
    self.params = parameters.Params()
    self.params.MAX_ENV_STEPS = max_steps
    self.physics_eng = physics.PhysicsSim()

    ## Ball XY positions can be between -1.5 and 1.5
    ## Arm joint can have positons:
    # Joint 0: [-Pi/2, Pi/2]
    # Joint 1: [-Pi, Pi]
    self.observation_space = spaces.Box(low=np.array([
                                          -self.params.TABLE_SIZE[0] / 2., -self.params.TABLE_SIZE[1] / 2., # ball pose
                                          -np.pi / 2, -np.pi,                                               # joint angles
                                          -50, -50]),                                                       # joint vels
                                        high=np.array([
                                          self.params.TABLE_SIZE[0] / 2., self.params.TABLE_SIZE[1] / 2.,   # ball pose
                                          np.pi / 2, np.pi,                                                 # joint angles
                                          50, 50]), dtype=np.float32)                                       # joint vels


    ## Joint commands can be between [-1, 1]
    self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32)

    self.goals = np.array([hole['pose'] for hole in self.physics_eng.holes])
    self.goalRadius = [hole['radius'] for hole in self.physics_eng.holes]
    self.rew_area = None

    self.seed(seed)

  def seed(self, seed=None):
    """
    Function to seed the environment
    :param seed: The random seed
    :return: [seed]
    """
    np.random.seed(seed)
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self, desired_ball_pose=None):
    """
    Function to reset the environment.
    - If param RANDOM_BALL_INIT_POSE is set, the ball appears in a random pose, otherwise it will appear at [-0.5, 0.2]
    - If param RANDOM_ARM_INIT_POSE is set, the arm joint positions will be set randomly, otherwise they will have [0, 0]
    :return: Initial observation
    """
    if self.params.RANDOM_BALL_INIT_POSE:
      init_ball_pose = np.array([self.np_random.uniform(low=-1.2, high=1.2),  # x
                                 self.np_random.uniform(low=-1.2, high=1.2)])  # y
    elif desired_ball_pose is not None:
      init_ball_pose = np.array(desired_ball_pose)
    else:
      init_ball_pose = np.array([-0.5, 0.2])

    if self.params.RANDOM_ARM_INIT_POSE:
      init_joint_pose = np.array([self.np_random.uniform(low=-np.pi * .2, high=np.pi * .2),  # Joint0
                                  self.np_random.uniform(low=-np.pi * .9, high=np.pi * .9)])  # Joint1
    else:
      init_joint_pose = None

    self.physics_eng.reset([init_ball_pose], init_joint_pose)
    self.steps = 0
    self.rew_area = None
    return self._get_obs()

  def _get_obs(self):
    """
    This function returns the state after reading the simulator parameters.
    :return: state: composed of ([ball_pose_x, ball_pose_y], [joint0_angle, joint1_angle], [joint0_speed, joint1_speed])
    """
    ball_pose = self.physics_eng.balls[0].position + self.physics_eng.wt_transform
    joint0_a = self.physics_eng.arm['jointW0'].angle
    joint0_v = self.physics_eng.arm['jointW0'].speed
    joint1_a = self.physics_eng.arm['joint01'].angle
    joint1_v = self.physics_eng.arm['joint01'].speed
    if np.abs(ball_pose[0]) > 1.5 or np.abs(ball_pose[1]) > 1.5:
      raise ValueError('Ball out of map in position: {}'.format(ball_pose))

    self.state = np.array([ball_pose[0], ball_pose[1], joint0_a, joint1_a, joint0_v, joint1_v])
    return self.state

  def reward_function(self):
    """
    This function calculates the reward
    :return:
    """
    ball_pose = self.state[0:2]
    for hole_idx, hole in enumerate(self.physics_eng.holes):
      dist = np.linalg.norm(ball_pose - hole['pose'])
      if dist <= hole['radius']:
        done = True
        reward = 100
        self.rew_area = hole_idx
        return reward, done
    return 0, False

  def step(self, action):
    """
    Performs an environment step.
    :param action: Arm Motor commands. Can be either torques or velocity, according to TORQUE_CONTROL parameter
    :return: state, reward, final, info
    """
    # action = np.clip(action, -1, 1)

    self.steps += 1
    ## Pass motor command
    self.physics_eng.move_joint('jointW0', action[0])
    self.physics_eng.move_joint('joint01', action[1])
    ## Simulate timestep
    self.physics_eng.step()
    ## Get state
    self._get_obs()

    # Get reward
    reward, done = self.reward_function()

    info = {}
    info['rew_area'] = self.rew_area
    if done: info['reason'] = 'Ball in hole'

    if self.steps >= self.params.MAX_ENV_STEPS:  ## Check if max number of steps has been exceeded
      done = True
      info['reason'] = 'Max Steps reached: {}'.format(self.steps)

    return self.state, reward, done, info

  def render(self, mode='rgb_array', **kwargs):
    """
    Rendering function
    :param mode: if human, renders on screen. If rgb_array, renders as numpy array
    :return: screen if mode=human, array if mode=rgb_array
    """
    # If no screen available create screen
    if self.screen is None and mode == 'human':
      self.screen = pygame.display.set_mode((self.params.DISPLAY_SIZE[0], self.params.DISPLAY_SIZE[1]), 0, 32)
      pygame.display.set_caption('Billiard')
      self.clock = pygame.time.Clock()

    if self.state is None: return None ## If there is no state, exit

    if mode == 'human':
      self.screen.fill(pygame.color.THECOLORS["white"]) ## Draw white background
    elif mode == 'rgb_array':
      capture = pygame.Surface((self.params.DISPLAY_SIZE[0], self.params.DISPLAY_SIZE[1]))
      capture.set_alpha(None)
      capture.fill((255, 255, 255))

    ## Draw holes. This are just drawn, but are not simulated.
    for goal, radius in zip(self.goals, self.goalRadius):
      ## To world transform (The - is to take into account pygame coordinate system)
      pose = np.array([goal[0], -goal[1]]) + self.physics_eng.tw_transform

      if mode == 'human':
        ## Draw the holes on the screen
        pygame.draw.circle(self.screen,
                           (255, 0, 0),
                           [int(pose[0] * self.params.PPM), int(pose[1] * self.params.PPM)],
                           int(radius * self.params.PPM))
      elif mode == 'rgb_array':
        ## Draw the holes on the capture
        pygame.draw.circle(capture,
                           (255, 0, 0),
                           [int(pose[0] * self.params.PPM), int(pose[1] * self.params.PPM)],
                           int(radius * self.params.PPM))

    ## Draw bodies
    for body in self.physics_eng.world.bodies:
      color = [0, 0, 0]
      obj_name = body.userData['name']
      if obj_name == 'ball0':
        color = [0, 0, 255]
      elif obj_name in ['link0', 'link1']:
        color = [100, 100, 100]
      elif 'wall' in obj_name:
        color = [150, 150, 150]

      for fixture in body.fixtures:
        if mode == 'human':
          fixture.shape.draw(body, self.screen, self.params, color)
        elif mode == 'rgb_array':
          obj_name = body.userData['name']
          if self.params.SHOW_ARM_IN_ARRAY: ## If param is set, in the rgb_array the arm will be visible
            fixture.shape.draw(body, capture, self.params, color)
          else:
            if not obj_name in ['link0', 'link1']:
              fixture.shape.draw(body, capture, self.params, color)

    if mode == 'human':
      pygame.display.flip() ## Need to flip cause of drawing reasons
      self.clock.tick(self.params.TARGET_FPS)
      return self.screen
    elif mode == 'rgb_array':
      imgdata = pygame.surfarray.array3d(capture)
      return imgdata.swapaxes(0, 1)
