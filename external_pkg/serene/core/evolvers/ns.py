# Created by Giuseppe Paolo 
# Date: 09/03/2020

from external_pkg.serene.core.evolvers import BaseEvolver
from external_pkg.serene.core.evolvers import utils


class NoveltySearch(BaseEvolver):
  """
  This class implements NS. It takes care of calculating novelty, adding agents to archive and select new population.
  """
  def __init__(self, parameters):
    super().__init__(parameters)
    self.update_criteria = 'novelty'

  def evaluate_performances(self, population, offsprings, pool=None):
    """
    This function evaluates the novelty of population and offsprings wrt pop+off+archive reference set.
    The novelty is evaluated according to the given distance metric
    :param population:
    :param offsprings:
    :param pool: Multiprocessing pool
    :return:
    """
    # Get BSs
    population_bd = population['bd']
    offsprings_bd = offsprings['bd']

    if self.archive.size > 0:
      archive_bd = self.archive['bd']
    else:
      archive_bd = []

    reference_set = population_bd + offsprings_bd + archive_bd
    bd_set = population_bd + offsprings_bd

    novelties = utils.calculate_novelties(bd_set, reference_set, distance_metric=self.params.novelty_distance_metric,
                                          novelty_neighs=self.params.novelty_neighs, pool=pool)
    # Update population and offsprings
    population['novelty'] = novelties[:population.size]
    offsprings['novelty'] = novelties[population.size:]





