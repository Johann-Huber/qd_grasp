# Created by Giuseppe Paolo 
# Date: 27/07/2020

from external_pkg.serene.core.controllers.base_controller import *

# ---------------------------------------------------------
class FFNeuralController(BaseController):
  """
  This class implements a small feedforward network
  """
  # ----------------------------------
  def __init__(self, input_size, output_size, hidden_layers=2, hidden_layer_size=5, bias=True, **kwargs):
    super(FFNeuralController, self).__init__()
    self.name = "FFNeural"
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_layer_size = hidden_layer_size
    self.hidden_layers = hidden_layers
    self.use_bias = bias

    if self.hidden_layers > 0:
      self.genome_size = self.input_size*self.hidden_layer_size + self.use_bias + \
                         (self.hidden_layer_size*self.hidden_layer_size + self.use_bias)*(self.hidden_layers-1) + \
                         self.hidden_layer_size*self.output_size + self.use_bias
    else:
      self.genome_size = self.input_size*self.output_size + self.use_bias
  # ----------------------------------

  # ----------------------------------
  def load_genome(self, genome):
    """
    Loads the genome
    :param genome: Genome as list of numbers
    :return:
    """
    assert len(genome) == self.genome_size, 'Wrong genome size. Expected {} - Given {}'.format(self.genome_size, len(genome))
    self.layers = []
    self.bias = []
    idx = 0
    if self.hidden_layers > 0:
      # Input to hidden
      start = idx
      end = start + self.input_size*self.hidden_layer_size
      idx = end
      layer = genome[start:end]
      self.layers.append(np.reshape(layer, (self.input_size, self.hidden_layer_size)))
      if self.use_bias:
        self.bias.append(genome[idx])
        idx += 1
      # Hidden to hidden
      for k in range(self.hidden_layers-1):
        start = idx
        end = start + self.hidden_layer_size*self.hidden_layer_size
        idx = end
        layer = genome[start:end]
        self.layers.append(np.reshape(layer, (self.hidden_layer_size, self.hidden_layer_size)))
        if self.use_bias:
          self.bias.append(genome[idx])
          idx += 1
      # Hidden to output
      start = idx
      end = start + self.hidden_layer_size * self.output_size
      idx = end
      layer = genome[start:end]
      self.layers.append(np.reshape(layer, (self.hidden_layer_size, self.output_size)))
      if self.use_bias:
        self.bias.append(genome[idx])
        idx += 1
    else:
      idx = self.input_size * self.output_size
      layer = genome[0:idx]
      self.layers.append(np.reshape(layer, (self.input_size, self.output_size)))
      if self.use_bias:
        self.bias.append(genome[idx])

    if self.use_bias:
      assert len(self.bias) == len(self.layers), 'Not enough bias or layers. Bias {} - Layers {}'.format(len(self.bias), len(self.layers))
  # ----------------------------------

  # ----------------------------------
  def evaluate(self, *args):
    """
    Evaluates agent
    :param args: Input of shape input_size
    :return: Output of the network
    """
    assert len(args) == 1, 'Too many inputs given to controller. Expected 1 - Given {}'.format(len(args))
    assert len(args[0]) == self.input_size, 'Wrong input size. Expected {} - Given {}'.format(self.input_size, len(args[0]))
    data = np.expand_dims(args[0], axis=0)

    for i in range(len(self.layers)):
      data = np.matmul(data, self.layers[i])
      if self.use_bias:
        data = self.bias[i] + data
      data = (expit(data)*2) - 1 # This way we translate the sigmoid in the [-1, 1] interval
    return data[0]
  # ----------------------------------
# ---------------------------------------------------------
