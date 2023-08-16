import numpy as np
# Params class
class Params(object):
  """
  Define simulation parameters.
  The world is centered at the lower left corner of the table.
  """
  def __init__(self):
    """
    Constructor
    """
    self.TABLE_SIZE = np.array([3., 3.])
    self.TABLE_CENTER = np.array(self.TABLE_SIZE / 2)
    self.DISPLAY_SIZE = (300, 300)
    self.TO_PIXEL = np.array(self.DISPLAY_SIZE) / self.TABLE_SIZE

    self.LINK_0_LENGTH = 1.
    self.LINK_1_LENGTH = 1.
    self.LINK_ELASTICITY = 0.
    self.LINK_FRICTION = .9
    self.LINK_THICKNESS = 0.05

    self.BALL_RADIUS = .15
    self.BALL_ELASTICITY = .9
    self.BALL_FRICTION = .9

    self.WALL_THICKNESS = .1
    self.WALL_ELASTICITY = .95
    self.WALL_FRICTION = .9

    self.VEL_ITER = 100
    self.POS_ITER = 100

  # Graphic params
    self.PPM = int(min(self.DISPLAY_SIZE)/max(self.TABLE_SIZE))
    self.TARGET_FPS = 60
    self.TIME_STEP = 1.0 / self.TARGET_FPS

    self.MAX_ENV_STEPS = 300

    self.TORQUE_CONTROL = False
    self.TEST = True

    self.RANDOM_ARM_INIT_POSE = False
    self.RANDOM_BALL_INIT_POSE = False

    self.SHOW_ARM_IN_ARRAY = True
