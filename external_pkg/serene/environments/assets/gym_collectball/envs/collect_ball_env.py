# Created by Giuseppe Paolo 
# Date: 27/08/2020
from gym_collectball.envs.pyFastSimEnv.DefaultNav_Env import *
import pyfastsim as fs
import numpy as np
import netpbmfile
import time
import os


class CollectBall(SimpleNavEnv):
  """
  2 Wheeled robot inside a maze, collecting balls and dropping them into a goal.
  The environment is an additional layer to pyFastSim.

  Default observation space is Box(10,) meaning 10 dimensions continuous vector
      0-2 are lasers oriented -45:0/45 degrees
      3-4 are left right bumpers
      5-6 are light sensors with angular ranges of 50 degrees, sensing balls represented as sources of light.
      7-8 are light sensors with angular ranges of 50 degrees, sensing goal also represented as a source of light.
      9 is indicating if a ball is held
  (edit the xml configuration file in ./pyFastSimEnv if you want to change the sensors)

  Action space is Box(3,) meaning 3 dimensions continuous vector, corresponding to the speed of the 2 wheels, plus
  a 'grabbing value', to indicate whether or not the robot should hold or release a ball. (<0 release, >0 hold)

  x,y are by default bounded in [0, 600].

  Fitness is the number of balls caught and released in the goal. The goal is by default placed at the initial
  robot position.

  Environment mutation corresponds to a translation of the balls + translation and rotation of the initial position
  of the robot at the start of one episode.
  """
  def __init__(self):
    super().__init__(os.path.dirname(__file__) + "/pyFastSimEnv/LS_maze_hard.xml")
    self.nb_ball = 6
    self.proximity_threshold = 10.0  # min distance required to catch or release ball
    self.initPos = (100, 500, 45)  # position is (x, y, rotation in degrees)
    self.rew_area = None
    posture = fs.Posture(*(100, 500, 45))
    self.robot.set_pos(posture)  # Set position in FastSim according to init_pos

    # Initial ball positions
    # self.init_balls = [(self.get_robot_pos()[0] + 60 * np.cos((2 * np.pi) * i / self.nb_ball),
    #                     self.get_robot_pos()[1] + 60 * np.sin((2 * np.pi) * i / self.nb_ball)) for i in range(self.nb_ball)]
    self.init_balls = [(75, 75), (75, 540), (225, 540), (450, 450), (450, 75), (225, 225)]

    self.action_space = spaces.Box(low=np.array([-self.maxVel]*2 + [-1.]), high=np.array([self.maxVel]*2 + [1.]), shape=(3,), dtype=np.float32)
    undefined_sensors = len(self.get_radars()) + len(self.get_light_sensors())
    n_lasers = len(self.robot.get_lasers())
    self.observation_space = spaces.Box(low=np.array([0.] * n_lasers + [0.] * 2 + [float("-inf")] * undefined_sensors + [0.]),
                                        high=np.array([self.maxSensorRange] * n_lasers + [1.] * 2 + [
                                          float("inf")] * undefined_sensors + [1.]),
                                        dtype=np.float32)
    self.goal_pos = self.get_robot_pos()
    self.ball_held = -1
    self.pos = (self.get_robot_pos()[0], self.get_robot_pos()[1])
    self.balls = {i:ball for i, ball in enumerate(self.init_balls)}
    self.collected_balls = []
    self.add_balls()
    self.windows_alive = True
    self.max_steps = 2000
    self.step_count = 0

  def reset(self):
    self.rew_area = None
    p = fs.Posture(*self.initPos)
    self.robot.set_pos(p)
    self.current_pos = self.get_robot_pos()
    self.v1_motor_order = 0.
    self.v2_motor_order = 0.
    self.step_count = 0
    self.balls = {i: ball for i, ball in enumerate(self.init_balls)}
    self.collected_balls = []
    self.add_balls()
    self.ball_held = -1
    return self.get_all_sensors() + [0.]

  def add_balls(self): # Clear stuff in the map and readds it. Used when the ball are removed or added
    self.map.clear_illuminated_switches()
    self.map.add_illuminated_switch(fs.IlluminatedSwitch(1, 8, self.goal_pos[0], self.goal_pos[1], True))
    for x, y in self.balls.values():
      self.map.add_illuminated_switch(fs.IlluminatedSwitch(0, 8, x, y, True))

  def catch(self):
    if self.ball_held == -1:
      for i in self.balls:
        if np.sqrt((self.pos[0] - self.balls[i][0]) ** 2 + (self.pos[1] - self.balls[i][1]) ** 2) < self.proximity_threshold:
          self.ball_held = i
          del self.balls[i]
          self.add_balls()
          return 0.0
    return 0.0

  def release(self):
    reward = 0.
    if self.ball_held != -1:
      # If collected we save its index in the collected ball list
      if np.sqrt((self.pos[0] - self.initPos[0]) ** 2 + (self.pos[1] - self.initPos[1]) ** 2) < self.proximity_threshold:
        self.collected_balls.append(self.ball_held)
        reward = 1.0
      else: # Else we drop it again on the ground
        self.balls[self.ball_held] = self.pos[:2]
        self.add_balls()
        # reward = -0.1
      self.ball_held = -1
    return reward

  def step(self, action):
    self.step_count += 1
    holding = action[2] > 0

    # Action is: [leftWheelVel, rightWheelVel]
    [v1, v2] = action[0] * 2. , action[1] * 2.

    self.v1_motor_order = np.clip(v1, -self.maxVel, self.maxVel)
    self.v2_motor_order = np.clip(v2, -self.maxVel, self.maxVel)

    self.robot.move(self.v1_motor_order, self.v2_motor_order, self.map, sticky_walls)

    sensors = self.get_all_sensors()

    self.old_pos = self.current_pos
    self.current_pos = self.get_robot_pos()
    if self.step_count >= self.max_steps: episode_over = True
    else: episode_over = False

    dist_obj = dist(self.current_pos, self.goalPos)

    reward = 0.0  # default reward is distance to goal, so we reset it to 0
    # Check ball and goal proximity only if necessary
    if holding:
      reward += self.catch()
    if not holding:
      reward += self.release()

    state = sensors
    state.append(1.0 if self.ball_held != -1 else 0.0)
    self.pos = (self.get_robot_pos()[0], self.get_robot_pos()[1])

    balls_pos = self.balls.copy()
    if self.ball_held != -1: # Add the position of the robot carrying the ball as position of the ball
      balls_pos[self.ball_held] = self.pos
    for collected in self.collected_balls:
      balls_pos[collected] = self.initPos[:2]
    balls_pos = [balls_pos[i] for i in range(len(balls_pos))]
    if reward > 0:
      self.rew_area = 0

    return state, reward, episode_over, {"dist_obj": dist_obj, "robot_pos": self.current_pos, "balls_pos": balls_pos, 'rew_area': self.rew_area}

  def render(self, mode='human', close=False):
    if mode == 'rgb_array':
      # Load background
      if 'maze_hard' in default_env:
        filepath = os.path.join(os.path.dirname(__file__), 'pyFastSimEnv/arena.pbm')
      else:
        filepath = os.path.join(os.path.dirname(__file__), 'pyFastSimEnv/cuisine.pbm')
      with open(filepath, 'rb') as f:
        background = netpbmfile.imread(f)
      # Get robot position
      robot_pos = self.get_robot_pos()
      robot_pos = np.array(robot_pos[:2]) * np.array(
        background.shape) / 600  # This is to scale the position to the image size
      robot_rad = 10
      # Draw robot
      xx, yy = np.mgrid[:background.shape[0], :background.shape[1]]
      circle = (xx - robot_pos[1]) ** 2 + (yy - robot_pos[0]) ** 2
      robot = np.array(circle < robot_rad ** 2, dtype=float)

      background = np.tile(np.expand_dims(np.clip(background, a_min=0., a_max=1.), -1), (1, 1, 3))  # Make it RGB
      robot = np.tile(np.expand_dims(np.clip(robot, a_min=0., a_max=1.), -1), (1, 1, 3)) * np.array(
        [0, 0, 1])  # Make it RGB with blue robot

      image = np.clip(background + robot, a_min=0., a_max=1.)
      # Get balls positions
      balls = []
      for ball in self.balls:
        xx, yy = np.mgrid[:background.shape[0], :background.shape[1]]
        ball_rad = 5
        ball = np.array(self.balls[ball]) * np.array(background.shape[:2]) / 600
        circle = (xx - ball[1]) ** 2 + (yy - ball[0]) ** 2
        ball = np.array(circle < ball_rad ** 2, dtype=float)
        ball = np.tile(np.expand_dims(np.clip(ball, a_max=1., a_min=0.), -1), (1, 1, 3)) * np.array([1, 0, 0])
        image = np.clip(image + ball, a_min=0., a_max=1.)

      return image

    if self.display:
      self.display.update()
      time.sleep(0.01)
    pass



