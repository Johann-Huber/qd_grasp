# Created by Giuseppe Paolo 
# Date: 08/09/2020
from external_pkg.serene.core.evolvers import BaseEvolver
from external_pkg.serene.core.evolvers import utils
import numpy as np
import itertools

class NSGAII(BaseEvolver):
  """
  This class implements the NSGA-II evolver.
  """
  def __init__(self, parameters):
    super(NSGAII, self).__init__(parameters)
    self.update_criteria = ['novelty', 'reward']

  def evaluate_performances(self, population, offsprings, pool=None):
    """
    This function evaluates the novelty of the population and off springs.
    Then calculates the front and does all the NSGA-II stuff related to the pareto front
    :param population:
    :param offsprings:
    :param pool:
    :return:
    """
    # Get BSs
    population_bd = population['bd']
    offsprings_bd = offsprings['bd']

    # Need the archive for the novelty
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

  def update_population(self, population, offsprings, generation):
    """
    This function updates the population by calculating the pareto front and the crowding distance
    :param population:
    :param offsprings:
    :param generation: Generation at which we are
    :return:
    """
    parents_off = population.pop + offsprings.pop
    performances = []
    for criteria in self.update_criteria:
      performances.append(population[criteria] + offsprings[criteria])
    non_dominated_sorted_solution = utils.fast_non_dominated_sort(*performances)

    # Get the fronts used to obtain new population
    total_len = 0
    fronts_used = []
    for front in non_dominated_sorted_solution:
      total_len = len(front) + total_len
      fronts_used.append(front)
      if total_len > len(population):
        break

    # Find elements in the last front to use in new population and add them to the list
    last_front = fronts_used.pop(-1)
    fronts_used = list(itertools.chain(*fronts_used)) # Flatten fronts as a list of indexes
    missing = len(population) - len(fronts_used)
    if missing > 0:
      crowding_distance_values = utils.crowding_distance(performances[0], performances[1], last_front)
      idx = np.argsort(crowding_distance_values)[::-1][:missing] # Get elements with highest distance from last front
      fronts_used += [last_front[i] for i in idx]

    for new_pop_idx, old_pop_idx in zip(range(population.size), fronts_used):
      population.pop[new_pop_idx] = parents_off[old_pop_idx]