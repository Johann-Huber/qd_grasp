import Box2D as b2
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from gym_billiard.utils import parameters


# TODO implement checks on balls spawning positions (not in holes or on arm or overlapped'

def draw_polygon(polygon, body, screen, params, color):
  """
  Function used to extend polygon shape with drawing function
  :param polygon: self
  :param body: body to draw
  :param screen: screen where to draw
  :param params:
  :param color:
  :return:
  """
  vertices = [(body.transform * v) * params.PPM for v in polygon.vertices]
  vertices = [(v[0], params.DISPLAY_SIZE[1] - v[1]) for v in vertices]
  pygame.draw.polygon(screen, color, vertices)

b2.b2.polygonShape.draw = draw_polygon

def my_draw_circle(circle, body, screen, params, color):
  """
  Function used to extend circle shape with drawing function
  :param circle: self
  :param body: body to draw
  :param screen: screen where to draw
  :param params:
  :param color:
  :return:
  """
  position = body.transform * circle.pos * params.PPM
  position = (position[0], params.DISPLAY_SIZE[1] - position[1])
  pygame.draw.circle(screen,
                     color,
                     [int(x) for x in position],
                     int(circle.radius * params.PPM))

b2.b2.circleShape.draw = my_draw_circle

class PhysicsSim(object):
  """
  Physics simulator
  """
  def __init__(self, balls_pose=[[0, 0]], arm_position=None, params=None):
    """
    Constructor
    :param balls_pose: Initial ball poses. Is a list of the ball poses [ball0, ball1, ...]
    :param arm_position: Initial arm position
    :param params: Parameters
    """
    if params is None:
      self.params = parameters.Params()
    else:
      self.params = params

    ## Physic simulator
    self.world = b2.b2World(gravity=(0, 0), doSleep=True)
    self.dt = self.params.TIME_STEP
    self.vel_iter = self.params.VEL_ITER
    self.pos_iter = self.params.POS_ITER
    self._create_table()
    self._create_balls(balls_pose)
    self._create_robotarm(arm_position)
    self._create_holes()

  def _create_table(self):
    """
    Creates the walls of the table
    :return:
    """
    ## Walls in world RF
    left_wall_body = self.world.CreateStaticBody(position=(0, self.params.TABLE_CENTER[1]),
                                                 userData={'name': 'left wall'},
                                                 shapes=b2.b2PolygonShape(box=(self.params.WALL_THICKNESS/2,
                                                                               self.params.TABLE_SIZE[1]/2)))

    right_wall_body = self.world.CreateStaticBody(position=(self.params.TABLE_SIZE[0], self.params.TABLE_CENTER[1]),
                                                  userData={'name': 'right wall'},
                                                  shapes=b2.b2PolygonShape(box=(self.params.WALL_THICKNESS/2,
                                                                                self.params.TABLE_SIZE[1] / 2)))

    upper_wall_body = self.world.CreateStaticBody(position=(self.params.TABLE_CENTER[0], self.params.TABLE_SIZE[1]),
                                                  userData={'name': 'upper wall'},
                                                  shapes=b2.b2PolygonShape(box=(self.params.TABLE_SIZE[0] / 2,
                                                                                self.params.WALL_THICKNESS/2)))
    bottom_wall_body = self.world.CreateStaticBody(position=(self.params.TABLE_CENTER[0], 0),
                                                   userData={'name': 'bottom wall'},
                                                   shapes=b2.b2PolygonShape(box=(self.params.TABLE_SIZE[0] / 2,
                                                                                 self.params.WALL_THICKNESS/2)))

    self.walls = [left_wall_body, upper_wall_body, right_wall_body, bottom_wall_body]

    ## world RF -> table RF
    self.wt_transform = -self.params.TABLE_CENTER
    ## table RF -> world RF
    self.tw_transform = self.params.TABLE_CENTER

  def _create_balls(self, balls_pose):
    """
    Creates the balls in the simulation at the given positions
    :param balls_pose: Initial pose of the ball in table RF
    :return:
    """
    ## List of balls in simulation
    self.balls = []

    for idx, pose in enumerate(balls_pose):
      pose = pose + self.tw_transform ## move balls in world RF
      ball = self.world.CreateDynamicBody(position=pose,
                                          bullet=True,
                                          allowSleep=False,
                                          userData={'name': 'ball{}'.format(idx)},
                                          linearDamping=1.1,
                                          angularDamping=2,
                                          fixtures=b2.b2FixtureDef(shape=b2.b2CircleShape(radius=self.params.BALL_RADIUS),
                                                                   density=1,
                                                                   friction=self.params.BALL_FRICTION,
                                                                   restitution=self.params.BALL_ELASTICITY,))
      self.balls.append(ball)

  def _calculate_arm_pose(self, arm_position=None):
    """
    This function calculates the arm initial position according to the joints angles in randiants
    :param arm_position: joint arm position in radians. The zero is at the vertical position
    :return: link0 and link1 position.
    """
    pose = {'link0_center': np.array((self.params.TABLE_CENTER[0], self.params.LINK_0_LENGTH/2)),
            'link1_center': np.array((self.params.TABLE_CENTER[0], self.params.LINK_0_LENGTH - .1 + self.params.LINK_1_LENGTH / 2)),
            'joint01_center': np.array([self.params.TABLE_CENTER[0], self.params.LINK_0_LENGTH]),
            'link0_angle':0,
            'link1_angle':0}
    if arm_position is not None:
      ## LINK 0
      l0_joint_center = np.array([self.params.TABLE_CENTER[0], 0]) # Get joint0 position
      pose['link0_center'] = pose['link0_center'] - l0_joint_center # Center link0 on joint0 position
      # Rotate link0 center according to joint0 angle
      x = np.cos(arm_position[0]) * pose['link0_center'][0] - np.sin(arm_position[0]) * pose['link0_center'][1]
      y = np.sin(arm_position[0]) * pose['link0_center'][0] + np.cos(arm_position[0]) * pose['link0_center'][1]
      pose['link0_center'] = np.array((x, y)) + l0_joint_center

      ## LINK 1
      l1_joint_center = pose['joint01_center'] # Get joint1 position
      pose['link1_center'] = pose['link1_center'] - l1_joint_center # Center link1 on joint1 position
      # Rotate link1 center according to joint1 angle
      x = np.cos(arm_position[1]) * pose['link1_center'][0] - np.sin(arm_position[1]) * pose['link1_center'][1]
      y = np.sin(arm_position[1]) * pose['link1_center'][0] + np.cos(arm_position[1]) * pose['link1_center'][1]

      l1_joint_center = l1_joint_center - l0_joint_center  # Center joint1 on joint0 position
      # Rotate joint1 center according to joint0 angle
      jx = np.cos(arm_position[0]) * l1_joint_center[0] - np.sin(arm_position[0]) * l1_joint_center[1]
      jy = np.sin(arm_position[0]) * l1_joint_center[0] + np.cos(arm_position[0]) * l1_joint_center[1]
      l1_joint_center = np.array((jx, jy)) + l0_joint_center

      pose['link1_center'] = np.array((x, y)) + l1_joint_center
      pose['joint01_center'] = l1_joint_center
      pose['link0_angle'] = arm_position[0]
      pose['link1_angle'] = arm_position[1]  # Have to also center the angle

    return pose

  def _create_robotarm(self, arm_position=None):
    """
    Creates the robotic arm.
    :param arm_position: Initial angular position
    :return:
    """
    arm_pose = self._calculate_arm_pose(arm_position)
    link0 = self.world.CreateDynamicBody(position=arm_pose['link0_center'],
                                         angle=arm_pose['link0_angle'],
                                         bullet=True,
                                         allowSleep=False,
                                         userData={'name': 'link0'},
                                         fixtures=b2.b2FixtureDef(
                                           shape=b2.b2PolygonShape(box=(self.params.LINK_THICKNESS,
                                                                        self.params.LINK_0_LENGTH/2)),
                                           density=5,
                                           friction=self.params.LINK_FRICTION,
                                           restitution=self.params.LINK_ELASTICITY))

    # The -.1 in the position is so that the two links can overlap in order to create the joint
    link1 = self.world.CreateDynamicBody(position=arm_pose['link1_center'],
                                         angle=arm_pose['link1_angle'],
                                         bullet=True,
                                         allowSleep=False,
                                         userData={'name': 'link1'},
                                         fixtures=b2.b2FixtureDef(
                                           shape=b2.b2PolygonShape(box=(self.params.LINK_THICKNESS,
                                                                        self.params.LINK_1_LENGTH / 2)),
                                           density=1,
                                           friction=self.params.LINK_FRICTION,
                                           restitution=self.params.LINK_ELASTICITY))

    jointW0 = self.world.CreateRevoluteJoint(bodyA=self.walls[3],
                                             bodyB=link0,
                                             anchor=self.walls[3].worldCenter,
                                             lowerAngle=-.4 * b2.b2_pi - arm_pose['link0_angle'],
                                             upperAngle=.4 * b2.b2_pi - arm_pose['link0_angle'],
                                             enableLimit=True,
                                             maxMotorTorque=100.0,
                                             motorSpeed=0.0,
                                             enableMotor=True)

    joint01 = self.world.CreateRevoluteJoint(bodyA=link0,
                                             bodyB=link1,
                                             anchor=arm_pose['joint01_center'],
                                             lowerAngle=-b2.b2_pi*0.9 + arm_pose['link0_angle'] - arm_pose['link1_angle'],
                                             upperAngle=b2.b2_pi*0.9 + arm_pose['link0_angle'] - arm_pose['link1_angle'],
                                             enableLimit=True,
                                             maxMotorTorque=100.0,
                                             motorSpeed=0.0,
                                             enableMotor=True)

    ## Arm definition with links and joints
    self.arm = {'link0': link0, 'link1': link1, 'joint01': joint01, 'jointW0': jointW0}

  def _create_holes(self):
    """
    Defines the holes in table RF. This ones are not simulated, but just defined as a list of dicts.
    :return:
    """
    # Holes in simulation. Represented as list of dicts.
    self.holes = [{'pose': np.array([-self.params.TABLE_SIZE[0] / 2, self.params.TABLE_SIZE[1] / 2]), 'radius': .4},
                  {'pose': np.array([self.params.TABLE_SIZE[0] / 2, self.params.TABLE_SIZE[1] / 2]), 'radius': .4}]

  def reset(self, balls_pose, arm_position):
    """
    Reset the world to the given arm and balls poses
    :param balls_pose:
    :param arm_position:
    :return:
    """
    ## Destroy all the bodies
    for body in self.world.bodies:
      if body.type is b2.b2.dynamicBody:
        self.world.DestroyBody(body)

    ## Recreate the balls and the arm
    self._create_balls(balls_pose)
    self._create_robotarm(arm_position)

  def move_joint(self, joint, value):
    """
    Move the given joint of the given value
    :param joint: Joint to move
    :param value: Speed or torque to add to the joint
    :return:
    """
    speed = self.arm[joint].motorSpeed
    if self.params.TORQUE_CONTROL:
      speed = speed + value * self.dt
    else:
      speed = value

    # Limit max joint speed
    self.arm[joint].motorSpeed = np.float(np.sign(speed)*min(1, np.abs(speed)))

  def step(self):
    """
    Performs a simulator step
    :return:
    """
    self.world.Step(self.dt, self.vel_iter, self.pos_iter)
    self.world.ClearForces()

if __name__ == "__main__":
  phys = PhysicsSim(balls_pose=[[0, 0], [1, 1]])
  print(phys.arm['link0'])
