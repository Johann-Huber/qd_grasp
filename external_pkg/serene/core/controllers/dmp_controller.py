# Created by Giuseppe Paolo 
# Date: 27/10/2020

from external_pkg.serene.core.controllers.base_controller import *

class DMPController(BaseController):
  """
  This class implements the DMP controller based on gaussians
  """
  # ----------------------------------
  def __init__(self, output_size, num_base_functions=5, **kwargs):
    super(DMPController, self).__init__()
    self.name = "DMPGauss"
    self.input_size = 1 # Always only the time as input size
    self.output_size = output_size
    self.num_base_functions = num_base_functions

    self.genome_size = self.output_size * self.num_base_functions
    self.centers = np.linspace(0, 1, self.num_base_functions)
    self.sigma = 0.15 # This control how much variation there can be in the output
    self.max_value = self._get_max_value()
  # ----------------------------------

  # ----------------------------------
  def _get_max_value(self):
    """
    This function calculates the maximum value the DMP can reach, so to scale its output in the [-1, 1] range.
    Need this function cause the max value changes according to the number of bases used
    :return:
    """
    x = np.linspace(0, 1, 50)
    return np.max([np.sum([self.gaussian(t, center) * w for center, w in zip(self.centers, np.ones(self.num_base_functions))]) for t in x])
  # ----------------------------------

  # ----------------------------------
  def load_genome(self, genome):
    """
    This function loads the genome
    :param genome:
    :return:
    """
    self.genome = np.reshape(genome, (self.output_size, self.num_base_functions))
  # ----------------------------------

  # ----------------------------------
  def gaussian(self, x, u):
    return 1/(self.sigma*np.sqrt(2*np.pi)) * np.exp(-((x - u)**2)/(2*self.sigma**2))
  # ----------------------------------

  # ----------------------------------
  def evaluate(self, *args):
    """
    This function evaluates the input
    :param args: Input of shape input size
    :return:
    """
    assert len(args) == 1, 'Too many inputs given to controller. Expected 1 - Given {}'.format(len(args))
    assert len(args[0]) == self.input_size, 'Wrong input size. Expected {} - Given {}'.format(self.input_size, len(args[0]))
    t = args[0]
    output = []
    for i in range(self.output_size):
      output.append(np.sum([self.gaussian(t, center) * w for center, w in zip(self.centers, self.genome[i])]))
    return np.array(output)/self.max_value
  # ----------------------------------

