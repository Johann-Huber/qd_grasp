# Created by Giuseppe Paolo 
# Date: 28/07/2020

from external_pkg.serene.core.controllers import BaseController

# ---------------------------------------------------------
class DummyController(BaseController):
  """
  This class implements a dummy controller that does nothing.
  Used for testing in the dummy environment
  """
  def __init__(self, output_size, **kwargs):
    super(DummyController, self).__init__()
    assert output_size == 2, 'Output must be 2D. Given {}'.format(output_size)
    self.output_size = output_size
    self.genome_size = output_size

  def load_genome(self, genome):
    self.genome = genome

  def evaluate(self, *args):
    return self.genome
# ---------------------------------------------------------

