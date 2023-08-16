# Created by Giuseppe Paolo 
# Date: 11/09/2020

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pygame as pg
from pygame.locals import *
import sys
from gym_redarm.envs import graphics
import time
import copy

#CONSTANTS
WHITE	= (255,255,255)
GRAY	= (225,225,225)
BLACK	= (0,0,0)
RED     = (255,50,50)
BLUE    = (50,50,255)
GREEN   = (50,255,50)

LINE_WIDTH = 3


class Parent(gym.Env):

  def __init__(self):
    super(Parent, self).__init__()
    # STATE
    self.q = None
    self.x = None
    self.q_goal = None
    self.x_goal = None
    self.x_Glob = None

    self.walls = []

    # GRAPHICS
    #  To use graphics you need a module 'Graphics.py', to not use graphics, use graphics = 0
    self.initiateGraphics(graphics=0, id_frame=0)

  def initiateGraphics(self, graphics, id_frame=0):
    self.graphics = graphics  # Use this canvas for plotting etc
    if graphics != 0:
      self.id_frame = id_frame
      self.displace = np.array(graphics.get_canvas_displacement(id_frame))  # This is where the plotting starts
      self.wall_displace = np.append(self.displace, np.array([0, 0]))
      self.screen = graphics.screen
      self.frameside = graphics.frameside * graphics.canvas_prop

  def dumpGraphics(self):

    if self.graphics != 0:
      self.id_frame = None
      self.displace = None
      self.wall_displace = None
      self.screen = None
      self.frameside = None

    self.graphics = 0

  def get_state(self):
    return [self.q, self.x]

  def set_goal(self, q_goal):
    self.q_goal = q_goal

  def step(self, action):
    pass

  def render(self, mode='human'):
    pass

  def reset(self):
    pass

  def drawAgent(self, *args, **kwargs):
    pass

  def drawCircle(self, x, col, rad=4, width=0):
    pos = self.displace + np.int32(x * self.frameside)
    pg.draw.circle(self.screen, col, pos, rad, width)

  def drawCross(self, x, col):
    pos = self.displace + np.int32(x * self.frameside)

    o1 = 5 * np.array([1, 1])
    o2 = 5 * np.array([-1, 1])

    pg.draw.line(self.screen, col, pos - o1, pos + o1, 4)
    pg.draw.line(self.screen, col, pos - o2, pos + o2, 4)

  def drawStar(self, x, col):
    pos = self.displace + np.int32(x * self.frameside)

    o1 = 6 * np.array([np.sqrt(3) / 2, .5])
    o2 = 6 * np.array([-np.sqrt(3) / 2, .5])
    o3 = 6 * np.array([0, 1])

    pg.draw.line(self.screen, col, pos - o1, pos + o1, 1)
    pg.draw.line(self.screen, col, pos - o2, pos + o2, 1)
    pg.draw.line(self.screen, col, pos - o3, pos + o3, 1)
    pg.draw.line(self.screen, col, pos + o1, pos - o1, 1)
    pg.draw.line(self.screen, col, pos + o2, pos - o2, 1)
    pg.draw.line(self.screen, col, pos + o3, pos - o3, 1)

  def drawLine(self, xi, xj, col):
    pos_i = self.displace + np.int32(xi * self.frameside)
    pos_j = self.displace + np.int32(xj * self.frameside)
    pg.draw.line(self.screen, col, pos_i, pos_j, 2)

  def drawSquare(self, c1, c2, col):
    """
    Draws a square
    :param c1: Corner 1
    :param c2: Corner 2
    :param col: Color
    :return:
    """
    c1 = self.displace[0] + np.int32(c1 * self.frameside)
    c2 = self.displace[1] + np.int32(c2 * self.frameside)
    p0 = np.int32([c1[0], c1[1]])
    p1 = np.int32([c1[0], c2[1]])
    p2 = np.int32([c2[0], c1[1]])
    p3 = np.int32([c2[0], c2[1]])

    pg.draw.line(self.screen, col, p0, p1, 1)
    pg.draw.line(self.screen, col, p1, p3, 1)
    pg.draw.line(self.screen, col, p2, p3, 1)
    pg.draw.line(self.screen, col, p0, p2, 1)

class ArmEnv(Parent):
  """
  Redundant arm environment. Developed by Pontus Loviken. You can find the original code here:
  https://github.com/Loviken/MCGB?fbclid=IwAR0fjENMPGVWg5s8ri7gr3t603j4zoDd3_oG11W_4t2O3fNHShmcuFSEqZI
  """
  def __init__(self, seed=None):
    """ Constructor
    :param seed: the random seed for the environment
    :param max_steps: the maximum number of steps the episode lasts
    :return:
    """
    super(ArmEnv, self).__init__()
    self.max_Steps = 250
    self.ts = 0
    self.seed(seed)
    self.task_type = 'multi' # multi or sequence. If multi there are multiple goal areas that can be reached. If sequence, it has to visit all of them

    # THE AGENT
    #  General settings
    self.dof = 20  # Degrees of freedom
    self.dim = [2, self.dof]  # [dim(task), dim(posture)]
    self.armLength = 1.0
    self.segLen = self.armLength / self.dof  # Length of an arm segment
    self.aMax = np.pi  # Max angle on joint (<= pi)
    self.wall_setting = 8
    self.update_steps = 2
    self.max_update_speed = 1.#/self.update_steps # This is the maximum speed we can get during one update. During the whole step, is 1
    self.resetting = False

    self.observation_space = spaces.Box(low=-np.ones(self.dof), high=np.ones(self.dof), dtype=np.float32)
    self.action_space = spaces.Box(low=-np.ones(self.dof), high=np.ones(self.dof), dtype=np.float32)

    #  Start conditions - this is overwritten if 'random_start' or 'm_start' is given.
    aStart = 1. * 4 * np.pi / self.dof
    self.angles = np.linspace(aStart, aStart / 4, self.dof)  # Start in a spiral
    self.joint_speeds = np.zeros_like(self.angles)
    self.angle_old = self.angles
    self.angle_drawn = self.angles  # For graphics
    self.root = np.array([0.5, 0.5])  # Position of arm's base
    self.angleStart = copy.deepcopy(self.angles)
    self.goals = np.array([[0.25, 0.25], [0.85, 0.85], [0.25, 0.90]])
    self.goalRadius = [0.05, 0.05, 0.05]
    self.reward_area = None

    # THE ENVIRONMENT
    cW = 0.2  # corridor width
    wW = 0.02  # wall thickness

    self.walls = np.array([[0, 0, 0, 0]])

    choice = self.wall_setting  # Add walls
    if choice == 0:
      self.walls = np.array([[.2, .2, .2, .2]])
    elif choice == 1:
      self.walls = np.array([[1 - cW - wW, 2 * cW, wW, 2 * cW],
                             [0., 4 * cW, 1 - cW, wW]])
    elif choice == 2:
      cW = 1. / 6
      self.walls = np.array([[cW, 1.5 * cW, wW, 1 - 3 * cW],
                             [1 - wW - cW, 1.5 * cW, wW, 1 - 3 * cW]])
    elif choice == 3:
      self.walls = np.array([[1.5 * cW, cW, 1 - 3 * cW, wW],
                             [1.5 * cW, 1 - wW - cW, 1 - 3 * cW, wW],
                             [cW, 1.5 * cW, wW, 1 - 3 * cW],
                             [1 - wW - cW, 1.5 * cW, wW, 1 - 3 * cW]])
    elif choice == 4:  # One u-shaped wall above
      self.walls = np.array([[1 - cW - wW, cW, wW, cW],
                             [cW, cW, 1 - 2 * cW, wW],
                             [cW, cW, wW, cW]])

    elif choice == 5:  # One u-shaped wall bellow
      self.walls = np.array([[cW, 1 - cW - wW, 1 - 2 * cW, wW],
                             [1 - cW - wW, 1 - 2 * cW, wW, cW],
                             [cW, 1 - 2 * cW, wW, cW]])

    elif choice == 6:  # Simple wall bellow
      self.walls = np.array([[2 * cW, 1 - cW - wW, cW, wW]])

    elif choice == 7:  # Long wall above
      self.walls = np.array([[cW, 1 - cW, 1 - 2 * cW, wW]])

    elif choice == 8:  # cross
      self.walls = np.array([[.5 - .5 * wW, 0, wW, cW],
                             [.5 - .5 * wW, 1 - cW, wW, cW],
                             [0, .5 - .5 * wW, cW, wW],
                             [1 - cW, .5 - .5 * wW, cW, wW]])

  def seed(self, seed=None):
    """
    Function to seed the environment
    :param seed: The random seed
    :return: [seed]
    """
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def get_state(self):
    q = self.ang2q(self.angles)
    obs = (q * 2.) - 1.
    return obs

  def step(self, action):
    """
    The action we send is the torque applied to the joints. This torque is applied for 5 internal simulation steps.
    The observation is the q (that is the normalized joint angles in [0, 1]), rescaled in the [-1, 1] range
    The end effector pose is given in the info.
    :param action: Is the torque applied to every joint on the arm
    :return:
    """
    self.resetting = False
    stepsLeft = self.update_steps
    stopped = False

    while stepsLeft > 0:
      dt = .01
      angle_next = action/2 * dt**2 + self.joint_speeds * dt + self.angles

      # If the update can be made
      if self.isLegal(self.ang2q(angle_next), self.ang2q(self.angles)):
        self.joint_speeds = (angle_next - self.angles)/dt
        self.angles = angle_next
        stepsLeft -= 1

      else:
        stopped = True
        stepsLeft = 0

    q = self.ang2q(self.angles)
    x = self.q2x(q)
    self.ts += 1

    done = stopped or self.ts >= self.max_Steps
    if done: reward = self.calculate_reward(x)
    else: reward = 0
    return self.get_state(), reward, done, {"End effector pose": x, 'rew_area': self.reward_area}

  def calculate_reward(self, x):
    if self.task_type == 'multi':
      for goal_idx, goal in enumerate(self.goals):
        dist = np.linalg.norm(x - goal)
        if dist <= self.goalRadius[goal_idx]:
          reward = (self.goalRadius[goal_idx] - dist) / self.goalRadius[goal_idx]
          self.reward_area = goal_idx
          return reward
    elif self.task_type == 'sequence':
      raise NotImplemented
    return 0

  def isLegal(self, q, q_prev):
    """
    Check that arm does not break the physics of the simulation.
      1: It can't move any joint a distance > maxStep in one transition
      2: It can't pass through walls, or move out of the arena
      3: The arm can't intersect itself.
    :param q:
    :param q_prev:
    :return:
    """
    joint_coord = self.getCoordinates(self.q2ang(q))

    # TEST 1: Would any joint travel too fast?
    if not self.resetting:
      joint_coord0 = self.getCoordinates(self.q2ang(q_prev))

      # Modified to only check end effector (for speed)
      speed = np.linalg.norm(joint_coord[-1] - joint_coord0[-1])/self.update_steps + self.joint_speeds[-1]

      if speed > self.max_update_speed:
        # print("Too fast")
        return False

    # TEST 2a: Is the arm within boundaries?
    if np.min(joint_coord) < 0 or np.max(joint_coord) >= 1:
      # print('Outside of arena')
      return False

    # TEST 2b: Is the arm within or crossing any walls?
    #  If dof is too low points will be too far away for collision test
    segLen = self.segLen

    if len(joint_coord) < 21:
      new_coord = np.zeros((21, 2))
      new_coord[:, 0] = np.interp(np.linspace(0, 1, 21), \
                                  np.linspace(0, 1, len(joint_coord)), joint_coord[:, 0])
      new_coord[:, 1] = np.interp(np.linspace(0, 1, 21), \
                                  np.linspace(0, 1, len(joint_coord)), joint_coord[:, 1])

      segLen *= self.dof * 1. / 21

      joint_coord = new_coord

    # Would it go through walls?
    walls = self.walls

    for i in range(len(walls)):
      for j in range(len(joint_coord)):
        x1 = walls[i, 0]
        x2 = x1 + walls[i, 2]
        y1 = walls[i, 1]
        y2 = y1 + walls[i, 3]

        if x1 < joint_coord[j, 0]  < x2 and y1 <  joint_coord[j, 1] < y2:
          return False

    # TEST 3: Would it go through itself?
    #  I will do this simpler by drawing a circle around every segment and see if two segments
    #  intersect
    radius = segLen * 0.95

    # Sort by x for faster clasification
    joint_coord = joint_coord[joint_coord[:, 0].argsort()]

    for i in range(len(joint_coord)):
      for j in range(i + 1, len(joint_coord)):
        dx = np.abs(joint_coord[i, 0] - joint_coord[j, 0])
        dy = np.abs(joint_coord[i, 1] - joint_coord[j, 1])

        # print([i,j,dx, radius])
        if dx > radius:
          break

        if dy + dx < radius:
          # print('Self collision')
          return False

    return True

  def reset(self):
    self.resetting = True # This one is used so the arm can move how it wants to get to the reset pose
    self.reward_area = None
    aStart = 1. * 4 * np.pi / self.dof
    self.angles = np.linspace(aStart, aStart / 4, self.dof)  # Start in a spiral
    self.joint_speeds = np.zeros_like(self.angles)
    self.angle_old = self.angles
    self.angle_drawn = self.angles  # For graphics
    self.root = np.array([0.5, 0.5])  # Position of arm's base
    self.angleStart = copy.deepcopy(self.angles)
    self.ts = 0
    return self.get_state()

  #  Update graphics
  #  Input:
  #	- x_goal 		Where the planner wants the end-effector to move next
  #	- background	A matrix in the background, for example the V-function
  def render(self, mode='human', x_goal=None, x_long_goal=None, background=None, legend=''):
    if self.graphics == 0:
      self.initiateGraphics(graphics=graphics.Basic(), id_frame=0)

    # Full update of background
    if False:  # self.stepsLeft == self.updateSteps or background is not None:
      if background is None:
        gph.draw_matrix(np.ones((1, 1)), self.id_frame, v_min=0, v_max=1, matrix_text=legend)
      else:
        v_min = np.min(background)
        v_max = np.max(background)
        v_min = 2 * v_min - v_max
        gph.draw_matrix(background, self.id_frame, v_min=v_min, v_max=v_max, matrix_text=legend)

      self.drawWalls()
    else:
      self.drawAgent(self.angle_drawn, WHITE)  # Paint over old

    # Draw arm and goal
    self.drawWalls()
    self.drawAgent(self.angles, BLACK)

    self.angle_drawn = self.angles

    for goal, radius in zip(self.goals, self.goalRadius):
      self.drawCircle(goal, RED, rad=np.int32(radius*self.frameside))

    if x_goal is not None:
      self.drawCircle(x_goal, RED)

    if x_long_goal is not None:
      self.drawCross(x_long_goal, BLUE)

    # pg.display.update()
    if mode == 'rgb_array':
      imgdata = pg.surfarray.array3d(self.screen)
      return imgdata.swapaxes(0,1)

  # GRAPHICAL METHODS
  def drawAgent(self, angles, col, line_width=LINE_WIDTH):
    # this will scale and draw a line between the coordinates
    # col is the color of the lines

    coord = self.getCoordinates(angles)
    pg.draw.lines(self.screen, col, False, self.displace + coord * self.frameside, line_width)

  def drawWalls(self):
    walls = self.walls

    for i in range(len(walls)):
      rect = self.wall_displace + walls[i] * self.frameside
      pg.draw.rect(self.screen, BLACK, rect)

  # Convert radians to posture q
  def ang2q(self, ang):
    return 1. * ang / (self.aMax * 2) + 0.5  # Change span from [-a,a] to [0,1]

  # Convert posture q to radians
  def q2ang(self, q):
    return (2. * q - 1) * self.aMax  # Change span from [0,1] to [-a,a]

  # Get x from q - Basically the forward model
  def q2x(self, q):
    coord = self.getCoordinates(self.q2ang(q))

    return coord[-1, :]

  # Transform angles of arm to coordinates of joints
  def getCoordinates(self, angles):
    coordX = np.zeros(self.dof + 1) + self.root[0]
    coordY = np.zeros(self.dof + 1) + self.root[1]
    tmpAng = np.pi / 2  # start angle

    for i in range(self.dof):
      tmpAng += angles[i]

      coordX[i + 1] = coordX[i] + self.segLen * np.cos(tmpAng)
      coordY[i + 1] = coordY[i] + self.segLen * np.sin(tmpAng)

    return np.array([coordX, coordY]).T

  ## PRIVATE METHODS ##
  # If a session is loaded new graphics need to be created to show graphics
  def add_graphics(self, graphics, id_frame):
    self.graphics = graphics
    self.displace = np.array(graphics.get_canvas_displacement(id_frame))  # This is where the plotting starts
    # self.wall_displace 	= np.append(self.displace , np.array([0,0]))
    self.screen = graphics.screen
    self.frameside = graphics.frameside * graphics.canvas_prop

  # To save an environment graphics must be discarded
  def remove_graphics(self):
    self.graphics = None
    self.displace = None
    # self.wall_displace 	= None
    self.screen = None
    self.frameside = None