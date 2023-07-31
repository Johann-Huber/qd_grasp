# Created by Giuseppe Paolo 
# Date: 09/03/2020

import numpy as np
from external_pkg.serene.core.population import Population
from external_pkg.serene.core.population import Archive

import utils.constants as consts

class BaseEvolver(object):
  """
  This class implements a base evolver that generates a population of offsprings given a population of parents
  """
  def __init__(self, parameters):
    """
    Constructor
    """
    #self.params = parameters
    self.sigma = parameters['mutation_parameters']['sigma']
    self.mu = parameters['mutation_parameters']['mu']
    self.indpb = parameters['mutation_parameters']['mut_prob']
    self.mutation_operator = np.random.normal
    self.archive = Archive(parameters)
    self.update_criteria = None  # ['reward', 'novelty', 'surprise']
    self.agent_template = None # self.params.agent_template # TODO REQUIRED PARAMS
    self.evaluated_points = 0  # This is the counter of the total number of agents ever generated/evaluated
    self.emitter_based = False
    self.emitters_data = {}
    self.emitters = {}
    self.emitter_candidate = {}
    self.evaluation_budget = parameters['evaluation_budget']

  def mutate_genome(self, parameters, genome):
    """
    This function mutates the genome by using the mutation operator.
    NB: The genome is clipped in the range [-1, 1]
    :param genome:
    :return:
    """
    mutation = [self.mutation_operator(self.mu, self.sigma) if np.random.random() < self.indpb else 0 for k in range(len(genome))]
    genome = genome + np.array(mutation)
    return genome.clip(parameters.genome_limit[0], parameters.genome_limit[1])

  def _generate_off(self, parameters, parent_id_gen):
    """
    This function generates the offsprings from a given parent
    :param parent_id_gen: A tuple containing (parent_id, parent_genome)
    :return: list of offsprings
    """
    offsprings = []
    for k in range(parameters.offsprings_per_parent):
      off = self.agent_template.copy() # Get new agent
      off['genome'] = self.mutate_genome(parent_id_gen[1]) # Add parent mutate genome
      off['parent'] = parent_id_gen[0] # Add parent ID
      off['ancestor'] = parent_id_gen[2] if parent_id_gen[2] is not None else parent_id_gen[0]
      offsprings.append(off)
    return offsprings

  def generate_offspring(self, parameters, parents, generation, pool=None):
    """
    This function generates the offspring from the population
    :return: Population of offsprings
    """
    offsprings = Population(parameters, init_size=0, name='offsprings')

    parent_genome = parents['genome']
    parent_ids = parents['id']
    parent_ancestor = parents['ancestor']

    if pool is not None:
      offs = pool.map(self._generate_off, zip(parent_ids, parent_genome, parent_ancestor))
    else:
      offs = []
      for id_gen in zip(parent_ids, parent_genome, parent_ancestor): # Generate offsprings from each parent
        offs.append(self._generate_off(id_gen))

    offsprings.pop = [off for p_off in offs for off in p_off]  # Unpack list of lists and add it to offsprings
    offs_ids = parents.agent_id + np.array(range(len(offsprings)))  # Calculate offs IDs
    offsprings['id'] = offs_ids  # Update offs IDs
    offsprings['born'] = [generation] * offsprings.size
    parents.agent_id = max(offs_ids) + 1 # This saves the maximum ID reached till now
    return offsprings

  def evaluate_performances(self, population, offsprings, pool=None):
    raise NotImplementedError("This needs to be implemented")

  def update_archive(self, parameters, population, offsprings, generation):
    """
    Updates the archive according to the strategy and the criteria given.
    :param population:
    :param offsprings:
    :return:
    """
    # Get list of ordered indexes according to selection strategy
    if parameters.selection_operator == 'random':
      idx = list(range(offsprings.size))
      np.random.shuffle(idx)
    elif parameters.selection_operator == 'best':
      performances = offsprings[self.update_criteria]
      idx = np.argsort(performances)[::-1]  # Order idx according to performances. (From highest to lowest)
    else:
      raise ValueError(
        'Please specify a valid selection operator for the archive. Given {} - Valid: ["random", "best"]'.format(
          parameters.selection_operator))

    # TODO This part gets slower with time.
    # Add to archive the first lambda offsprings in the idx list
    for i in idx[:parameters._lambda]:
      offsprings[i]['stored'] = generation
      self.archive.store(offsprings[i])

  def update_population(self, population, offsprings, generation, id_counter):
    """
    This function updates the population according to the given criteria
    :param population:
    :param offsprings:
    :return:
    """
    performances = population[self.update_criteria] + offsprings[self.update_criteria]
    idx = np.argsort(performances)[::-1]  # Order idx according to performances.
    parents_off = population.pop + offsprings.pop
    # Update population list by going through it and putting an agent from parents+off at its place
    for new_pop_idx, old_pop_idx in zip(range(population.size), idx[:population.size]):
      population.pop[new_pop_idx] = parents_off[old_pop_idx]

  def init_emitters(self, parameters, ns_pop, ns_off, id_counter):
    return False