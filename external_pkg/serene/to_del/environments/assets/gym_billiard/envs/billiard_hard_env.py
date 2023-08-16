import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_billiard.utils import physics, parameters
import logging
logger = logging.getLogger(__name__)

# TODO implement logger

class BilliardHardEnv(gym.Env):
  """
  The goal is to send first ball 1 and then ball 0 in the holes. If the arm touches ball 1, is considered as failure.
    State is composed of:
    s = ([ball0_x, ball0_y], [ball1_x, ball1_y], [joint0_angle, joint1_angle], [joint0_speed, joint1_speed])

    The values that these components can take are:
    ball_x, ball_y -> [-1.5, 1.5]
    joint0_angle -> [-pi/2, pi/2]
    joint1_angle -> [-pi, pi]
    joint0_speed, joint1_speed -> [-50, 50]
    """
  metadata = {'render.modes': ['human'],
              'video.frames_per_second':15
              }

  def __init__(self, seed=None, max_steps=500):
    """ Constructor
    :param seed: the random seed for the environment
    :param max_steps: the maximum number of steps the episode lasts
    :return:
    """
    self.screen = None
    self.params = parameters.Params()
    self.physics_eng = physics.PhysicsSim()
    self.params.MAX_ENV_STEPS = max_steps

    ## Ball XY positions can be between -1.5 and 1.5
    ball0_os = spaces.Box(low=np.array([-self.params.TABLE_SIZE[0]/2., -self.params.TABLE_SIZE[1]/2.]),
                          high=np.array([self.params.TABLE_SIZE[0] / 2., self.params.TABLE_SIZE[1] / 2.]))

    ball1_os = spaces.Box(low=np.array([-self.params.TABLE_SIZE[0]/2., -self.params.TABLE_SIZE[1]/2.]),
                          high=np.array([self.params.TABLE_SIZE[0]/2., self.params.TABLE_SIZE[1]/2.]))

    ## Arm joint can have positons:
    # Joint 0: [-Pi/2, Pi/2]
    # Joint 1: [-Pi, Pi]
    joints_angle = spaces.Box(low=np.array([-np.pi / 2, -np.pi]), high=np.array([np.pi / 2, np.pi]))
    joints_vel = spaces.Box(low=np.array([-50, -50]), high=np.array([50, 50]))

    self.observation_space = spaces.Tuple([ball0_os, ball1_os, joints_angle, joints_vel])

    ## Joint commands can be between [-1, 1]
    self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]))

    self.seed(seed)

  def seed(self, seed=None):
    """
    Function to seed the environment
    :param seed: The random seed
    :return: [seed]
    """
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    """
    Function to reset the environment.
    - If param RANDOM_BALL_INIT_POSE is set, the ball appears in a random pose, otherwise ball0 will appear at [-0.5, 0.2] and ball1 at [0, 1]
    - If param RANDOM_ARM_INIT_POSE is set, the arm joint positions will be set randomly, otherwise they will have [0, 0]
    :return: Initial observation
    """
    if self.params.RANDOM_BALL_INIT_POSE:
      init_ball0_pose = np.array([self.np_random.uniform(low=-1.3, high=1.3), # x
                                 self.np_random.uniform(low=-1.3, high=0)])  # y
      init_ball1_pose = np.array([self.np_random.uniform(low=-1, high=1),  # x
                                  self.np_random.uniform(low=1, high=1.3)])  # y
    else:
      init_ball0_pose = np.array([-0.5, 0.2])
      init_ball1_pose = np.array([0, 1])

    if self.params.RANDOM_ARM_INIT_POSE:
      init_joint_pose = np.array([self.np_random.uniform(low=-np.pi * .2, high=np.pi * .2),  # Joint0
                                  self.np_random.uniform(low=-np.pi * .9, high=np.pi * .9)])  # Joint1
    else:
      init_joint_pose = None

    self.physics_eng.reset([init_ball0_pose, init_ball1_pose], init_joint_pose)
    self.steps = 0
    self.ball0_in_hole = False
    self.ball1_in_hole = False
    return self._get_obs()

  def _get_obs(self):
    """
    This function returns the state after reading the simulator parameters.
    :return: state: composed of ([ball0_pose_x, ball0_pose_y], [ball1_pose_x, ball1_pose_y], [joint0_angle, joint1_angle], [joint0_speed, joint1_speed])
    """
    ball0_pose = self.physics_eng.balls[0].position + self.physics_eng.wt_transform
    ball1_pose = self.physics_eng.balls[1].position + self.physics_eng.wt_transform

    if np.abs(ball0_pose[0])> 1.5 or np.abs(ball0_pose[1]) > 1.5:
      raise ValueError('Ball 0 out of map in position: {}'.format(ball0_pose))
    if np.abs(ball1_pose[0])> 1.5 or np.abs(ball1_pose[1]) > 1.5:
      raise ValueError('Ball 1 out of map in position: {}'.format(ball1_pose))

    joint0_a = self.physics_eng.arm['jointW0'].angle
    joint0_v = self.physics_eng.arm['jointW0'].speed
    joint1_a = self.physics_eng.arm['joint01'].angle
    joint1_v = self.physics_eng.arm['joint01'].speed
    self.state = (np.array([ball0_pose[0], ball0_pose[1]]),
                  np.array([ball1_pose[0], ball1_pose[1]]),
                  np.array([joint0_a, joint1_a]),
                  np.array([joint0_v, joint1_v]))
    return self.state

  def arm_ball1_contacts(self):
    """
    This function checks what is having contacts with ball1.
    :return: True if arm touches the ball
    """
    for contact in self.physics_eng.balls[1].contacts:
      if 'link' in contact.other.userData['name'] and contact.contact.touching:
        return True
    return False

  def step(self, action):
    """
    Performs an environment step.
    :param action: Arm Motor commands. Can be either torques or velocity, according to TORQUE_CONTROL parameter
    :return: state, reward, final, info
    """
    # action = np.clip(action, -1, 1)

    # Pass motor command
    self.physics_eng.move_joint('jointW0', action[0])
    self.physics_eng.move_joint('joint01', action[1])
    # Simulate timestep
    self.physics_eng.step()
    # Get state
    self._get_obs()
    reward = 0

    final = False
    # Check if final state
    # Calculates if distance between the ball's center and the holes' center is smaller than the holes' radius
    ball0_pose = self.state[0]
    ball1_pose = self.state[1]
    info = {}
    # If arm touches ball 1 finish episode
    if self.arm_ball1_contacts():
      reward = -100
      final = True
      info['reason'] = 'Arm touched ball 1'
    else:
      for hole in self.physics_eng.holes:
        dist_ball0 = np.linalg.norm(ball0_pose - hole['pose'])
        dist_ball1 = np.linalg.norm(ball1_pose - hole['pose'])

        if dist_ball0 <= hole['radius']:
          self.ball0_in_hole = True
        if dist_ball1 <= hole['radius']:
          self.ball1_in_hole = True

        ## Only B1 in hole, get reward but keep on playing
        if self.ball1_in_hole and not self.ball0_in_hole:
          final = False
          reward = 50
        ## Both in hole, get reward and finish the game
        elif self.ball1_in_hole and self.ball0_in_hole:
          final = True
          reward = 50
          info['reason'] = 'Both balls in holes'
        ## Only B0 in hole, get negative reward and finish the game
        elif not self.ball1_in_hole and self.ball0_in_hole:
          final = True
          reward = -50
          info['reason'] = 'Ball 0 in hole before ball 1'

    self.steps += 1
    if self.steps >= self.params.MAX_ENV_STEPS: ## Check if max number of steps has been exceeded
      final = True
      info['Reason'] = 'Max Steps reached: {}'.format(self.steps)

    return self.state, reward, final, info

  def render(self, mode='human'):
    """
    Rendering function
    :param mode: if human, renders on screen. If rgb_array, renders as numpy array
    :return: screen if mode=human, array if mode=rgb_array
    """
    import pygame

    # If no screen available create screen
    if self.screen is None and mode=='human':
      self.screen = pygame.display.set_mode((self.params.DISPLAY_SIZE[0], self.params.DISPLAY_SIZE[1]), 0, 32)
      pygame.display.set_caption('Billiard')
      self.clock = pygame.time.Clock()

    if self.state is None: return None ## If there is no state, exit

    if mode=='human':
      self.screen.fill(pygame.color.THECOLORS["white"]) ## Draw white background
    elif mode=='rgb_array':
      capture = pygame.Surface((self.params.DISPLAY_SIZE[0], self.params.DISPLAY_SIZE[1]))

    ## Draw holes. This are just drawn, but are not simulated.
    for hole in self.physics_eng.holes:
      ## To world transform (The - is to take into account pygame coordinate system)
      pose = -hole['pose'] + self.physics_eng.tw_transform

      if mode=='human':
        ## Draw the holes on the screen
        pygame.draw.circle(self.screen,
                           (255, 0, 0),
                           [int(pose[0] * self.params.PPM), int(pose[1] * self.params.PPM)],
                           int(hole['radius'] * self.params.PPM))
      elif mode=='rgb_array':
        ## Draw the holes on the capture
        pygame.draw.circle(capture,
                           (255, 0, 0),
                           [int(pose[0] * self.params.PPM), int(pose[1] * self.params.PPM)],
                           int(hole['radius'] * self.params.PPM))

    ## Draw bodies
    for body in self.physics_eng.world.bodies:
      color = [0, 0, 0]
      obj_name = body.userData['name']
      if obj_name == 'ball0':
        color = [0, 0, 180, 255]
      elif obj_name == 'ball1':
        color = [0, 180, 0, 255]
      elif obj_name in ['link0', 'link1']:
        color = [100, 100, 100]
        if mode=='rgb_array':
          color = [0, 0, 0]
      elif 'wall' in obj_name:
        color = [150, 150, 150]

      for fixture in body.fixtures:
        if mode=='human':
          fixture.shape.draw(body, self.screen, self.params, color)
        elif mode=='rgb_array':
          obj_name = body.userData['name']
          if self.params.SHOW_ARM_IN_ARRAY:  ## If param is set, in the rgb_array the arm will be visible
            fixture.shape.draw(body, capture, self.params, color)
          else:
            if not obj_name in ['link0', 'link1']:
              fixture.shape.draw(body, capture, self.params, color)

    if mode=='human':
      pygame.display.flip() ## Need to flip cause of drawing reasons
      self.clock.tick(self.params.TARGET_FPS)
      return self.screen
    elif mode=='rgb_array':
      imgdata = pygame.surfarray.array3d(capture)
      return imgdata.swapaxes(0, 1)