# Created by Giuseppe Paolo 
# Date: 27/07/2020

import numpy as np
from scipy.special import expit

# ---------------------------------------------------------
class BaseController(object):
  """
  This class implements the base controller from which other controllers inherit.
  """
  def __init__(self):
    """
    Constructor
    """
    self.name = "BaseController"
    self.genome = None

  def load_genome(self, genome):
    """
    This function loads the genome so to apply it
    :param genome: Genome to load
    :return:
    """
    self.genome = genome

  def evaluate(self, *args):
    """
    This function evaluates the genome on the given input
    :param x: Input
    :return:
    """
    raise NotImplementedError

  def __call__(self, *args):
    """
    This function calls the evaluate function
    :param args: input
    :return:
    """
    return self.evaluate(*args)
# ---------------------------------------------------------