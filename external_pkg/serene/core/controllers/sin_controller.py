# Created by Giuseppe Paolo 
# Date: 29/09/2020

from external_pkg.serene.core.controllers.base_controller import *

# ---------------------------------------------------------
class SinController(BaseController):
  """
  This class implements an oscillator controller
  """
  # ----------------------------------
  def __init__(self, input_size, output_size, name='SinContr', **kwargs):
    super(SinController, self).__init__()
    self.name = name
    self.input_size = input_size
    self.output_size = output_size
    self.degree = 3
    self.genome_size = 3 * self.output_size
  # ----------------------------------

  # ----------------------------------
  def sin_control(self, x):
    """
    Sinusoidal "oscillator". It has a sinusoidal oscillator for each degree of freedom.

    :param x:
    :return:
    """
    for d in range(self.degree):
      arg = self.params[0] * x + self.params[2]
      x = self.params[1] * np.sin(arg)
    return x
  # ----------------------------------

  # ----------------------------------
  def evaluate(self, *args):
    assert len(args) == 1, 'Too many inputs given to controller. Expected 1 - Given {}'.format(len(args))
    data = args[0]  # This controller work directly on the time.
    return self.sin_control(data)
  # ----------------------------------

  # ----------------------------------
  def load_genome(self, genome):
    """
    Load the genome
    :param genome:
    :return:
    """
    self.params = [] # Params is a list of lists. Every list is a set of (w, A, phi) of each oscillator
    start = 0
    end = 3
    for i in range(self.output_size):
      w, A, phi = genome[start:end] # In the [-1, 1] interval
      A = (A + 1) / 2 # Recenter in [0, 1]
      w = np.pi * (w + 3) / 2 # Recenter in [pi, 2*pi]
      self.params.append([w, A, phi])
      start = end
      end += 3
    self.params = np.array(self.params).transpose() # shape: [3, output_shape]
  # ----------------------------------
# ---------------------------------------------------------