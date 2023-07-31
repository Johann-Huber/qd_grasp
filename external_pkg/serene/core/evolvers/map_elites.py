# Created by Giuseppe Paolo 
# Date: 17/12/2020

from external_pkg.serene.core.evolvers import BaseEvolver
from external_pkg.serene.core.population.archive import Grid
from external_pkg.serene.core.population.population import Population
from external_pkg.serene.environments.environments import registered_envs
import numpy as np

class MAPElites(BaseEvolver):
  """
  This class implements an interface for MAP-Elites
  """
  def __init__(self, parameters):
    super().__init__(parameters)
    grid_params = registered_envs[self.params.env_name]['grid']
    self.archive = Grid(parameters, grid_params)
    self.initial_pop = True

  def generate_offspring(self, parents, generation, pool=None):
    # Do this only the first time, when the first N agents are generated
    if self.initial_pop:
      for agent in parents:
        self.archive.store(agent)
      self.initial_pop = False

    offsprings = Population(self.params, init_size=0, name='offsprings')
    for i in range(self.params.pop_size): # The batch is the pop size
      parent = self.archive[np.random.choice(self.archive.size)]
      off = self.agent_template.copy() # Get new agent
      off['genome'] = self.mutate_genome(parent[self.params.archive_stored_info.index('genome')])
      off['parent'] = parent[self.params.archive_stored_info.index('id')]
      if parent[self.params.archive_stored_info.index('ancestor')] is not None:
        off['ancestor'] = parent[self.params.archive_stored_info.index('ancestor')]
      else:
        off['ancestor'] = parent[self.params.archive_stored_info.index('id')]
      offsprings.add(off)

    offs_ids = parents.agent_id + np.array(range(len(offsprings)))  # Calculate offs IDs
    offsprings['id'] = offs_ids  # Update offs IDs
    offsprings['born'] = [generation] * offsprings.size
    parents.agent_id = max(offs_ids) + 1  # This saves the maximum ID reached till now
    return offsprings

  def evaluate_performances(self, population, offsprings, pool=None):
    """Nothing to evaluate here"""
    return

  def update_population(self, population, offsprings, generation):
    """Nothing to update here"""
    return

  def update_archive(self, population, offsprings, generation):
    """
    Here we add the evaluated offsprings to the archive
    :param population:
    :param offsprings:
    :param generation:
    :return:
    """
    for agent in offsprings:
      agent['stored'] = generation
      self.archive.store(agent)