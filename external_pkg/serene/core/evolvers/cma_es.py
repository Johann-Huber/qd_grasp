# Created by Giuseppe Paolo 
# Date: 29/07/2020

import numpy as np
from cmaes import CMA
from external_pkg.serene.core.evolvers import BaseEvolver
from external_pkg.serene.core.population import Population, Archive
from external_pkg.serene.environments.environments import registered_envs

class CMAES(BaseEvolver):
  """
  This class is a wrapper around the CMA ES implementation.
  """
  def __init__(self, parameters):
    super().__init__(parameters)
    self.update_criteria = 'fitness'
    self.sigma = 0.05

    # Instantiated only to extract genome size
    controller = registered_envs[self.params.env_name]['controller']['controller'](
      input_size=registered_envs[self.params.env_name]['controller']['input_size'],
      output_size=registered_envs[self.params.env_name]['controller']['output_size'])

    self.genome_size = controller.genome_size
    self.bounds = self.params.genome_limit * np.ones((self.genome_size, len(self.params.genome_limit)))
    self.values = []

    self.optimizer = CMA(mean=self.mu * np.ones(self.genome_size),
                         sigma=self.sigma,
                         bounds=self.bounds,
                         seed=self.params.seed,
                         population_size=self.params.emitter_population
                         )
    self.restarted_at = 0

  def generate_offspring(self, parents, generation, pool=None):
    """
    This function returns the parents. This way the population is evaluated given that contrary to the other algos
    here the population is given by the CMA-ES library
    :param parents:
    :param pool:
    :return:
    """
    return parents

  def evaluate_performances(self, population, offsprings, pool=None):
    """
    This function evaluates performances of the population. It's what calls the tell function from the optimizer
    The novelty is evaluated according to the given distance metric
    :param population:
    :param offsprings:
    :param pool: Multiprocessing pool
    :return:
    """
    solutions = [(genome, -value) for genome, value in zip(population['genome'], population['reward'])]
    self.values += [-val[1] for val in solutions]
    self.optimizer.tell(solutions)

  def check_stopping_criteria(self, generation):
    """
    This function is used to check for when to stop the emitter
    :param emitter_idx:
    :return:
    """
    if self.optimizer.should_stop():
      return True
    elif self._stagnation(generation-self.restarted_at):
      return True
    else: return False

  def _stagnation(self, cma_es_step):
    """
    Calculates the stagnation criteria
    :param emitter_idx:
    :param ca_es_step:
    :return:
    """
    bottom = int(20*self.genome_size/self.params.emitter_population + 120 + 0.2*cma_es_step)
    if cma_es_step > bottom:
      values = self.values[-bottom:]
      if np.median(values[:20]) >= np.median(values[-20:]) or np.max(values[:20]) >= np.max(values[-20:]):
        return True
    return False

  def update_archive(self, population, offsprings, generation):
    """
    Updates the archive. In this case the archive is a copy of the population.
    We do not really have the concept of archive in CMA-ES, so this archive here is just for ease of analysis and
    code compatibility.
    :param population:
    :param offsprings:
    :return:
    """
    # del self.archive
    # self.archive = Archive(self.params)

    for i in range(population.size):
      population[i]['stored'] = generation
      self.archive.store(population[i])

  def update_population(self, population, offsprings, generation):
    """
    This function updates the population according to the given criteria. For CMA-ES we use the ask function of the
    library
    :param population:
    :param offsprings:
    :return:
    """
    # In case a stopping criteria has been met, reinitialize the optimizer with the best agent in the archive (that is
    # the best found so far)
    if self.check_stopping_criteria(generation):
      print("Restarting")
      best = np.argmax(self.archive['reward'])
      self.restarted_at = generation

      self.values = []
      genome_idx = self.params.archive_stored_info.index('genome')
      self.optimizer = CMA(mean=self.archive[best][genome_idx],
                           sigma=self.sigma,
                           bounds=self.bounds,
                           seed=self.params.seed,
                           population_size=self.params.emitter_population
                           )

    population.empty()
    for idx in range(self.params.emitter_population):
      population.add()
      population[idx]['genome'] = self.optimizer.ask()
      population[idx]['born'] = generation
