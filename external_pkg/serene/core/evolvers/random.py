# Created by Giuseppe Paolo 
# Date: 11/01/2021

from external_pkg.serene.core.evolvers import BaseEvolver
from external_pkg.serene.core.population import Population
from external_pkg.serene.core.evolvers import utils

class RandomSearch(BaseEvolver):
  """
  This class implements Random search. At every generation it randomly samples a new population
  """
  def __init__(self, parameters):
    super().__init__(parameters)
    self.params.selection_operator = 'random'

  def generate_offspring(self, parents, generation, pool=None):
    """
    This function generates the offspring from the population
    :return: Population of offsprings
    """
    offsprings = Population(self.params, init_size=self.params.pop_size, name='offsprings')
    return offsprings

  def evaluate_performances(self, population, offsprings, pool=None):
    """ Nothing to evaluate here """
    return

  def update_population(self, population, offsprings, generation):
    """
    This function updates the population according to the given criteria
    :param population:
    :param offsprings:
    :return:
    """
    # Update population by copying offs in it
    population.pop = offsprings.pop.copy()





