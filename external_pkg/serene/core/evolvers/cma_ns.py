# Created by Giuseppe Paolo 
# Date: 30/07/2020

from external_pkg.serene.core.evolvers import EmitterEvolver
from external_pkg.serene.core.evolvers import utils
from external_pkg.serene.core.population import Population, Archive
from cmaes import CMA
import numpy as np
import copy
from external_pkg.serene.analysis.logger import Logger
import utils.constants as consts

class CMAESEmitter(object):
  """
  This class is a wrapper for the CMA-ES algorithm
  """
  def __init__(self, ancestor, mutation_rate, bounds, parameters):
    self.ancestor = ancestor
    self._init_mean = self.ancestor['genome']
    self.id = self.ancestor['id']
    self._mutation_rate = mutation_rate
    self.steps = 0
    self._params = parameters
    self._bounds = bounds
    self.pop = Population(self._params, init_size=self._params.emitter_population, name='cma_es')
    self.ns_arch_candidates = Population(self._params, init_size=0, name='ns_arch_cand')

    # List of lists. Each inner list corresponds to the values obtained during a step
    # We init with the ancestor reward so it's easier to calculate the improvement
    self.values = []
    self.archived_values = []
    self.improvement = 0
    self._init_values = None
    self.archived = [] # List containing the number of archived agents at each step
    self.most_novel = self.ancestor

    self._cmaes = CMA(mean=self._init_mean.copy(),
                      sigma=self._mutation_rate,
                      bounds=self._bounds,
                      seed=self._params.seed,
                      population_size=self._params.emitter_population)

  def estimate_improvement(self):
    """
    This function calculates the improvement given by the last updates wrt the parent
    If negative improvement, set it to 0.
    If there have been no updates yet, return the ancestor parent as reward
    Called at the end of the emitter evaluation cycle
    :return:
    """
    if self._init_values is None:  # Only needed at the fist time
      self._init_values = self.values[:3]
    self.improvement = np.max([np.mean(self.values[-3:]) - np.mean(self._init_values), 0])

    # If local improvement update init_values to have improvement calculated only on the last expl step
    if self._params.local_improvement:
      self._init_values = self.values[-3:]

  def ask(self):
    return self._cmaes.ask()

  def tell(self, solutions):
    return self._cmaes.tell(solutions)

  def should_stop(self):
    """
    Checks internal stopping criteria
    :return:
    """
    return self._cmaes.should_stop()


class CMANS(EmitterEvolver):
  """
  This class implements the CMA-NS evolver. It performs NS till a reward is found, then uses CMA-ES to search the reward
  area.
  """
  def create_emitter(self, parameters, parent_id, ns_pop, ns_off, id_counter):
    """
    This function creates the emitter
    :param parent:
    :param ns_pop:
    :param ns_off:
    :return:
    """
    return CMAESEmitter(ancestor=self.rewarding[parent_id].copy(),
                        mutation_rate=self.calculate_init_sigma(ns_pop, ns_off, self.rewarding[parent_id]),
                        bounds=self.bounds,
                        parameters=self.params)

  def update_cmaes_population(self, emitters, emitter_idx, generation):
    """
    This function returns the CMA-ES population to be evaluated.
    Everytime is called it reinitializes the evolver population with the one given by the chosen emitter.
    This because the evolver keep only one population structure that contains the population of the emitter being
    evaluated at the moment.
    :param emitter_idx: Index of the emitter in the emitter list
    :return:
    """
    emitters[emitter_idx].pop['genome'] = [emitters[emitter_idx].ask() for i in range(self.params.emitter_population)]
    emitters[emitter_idx].pop['parent'] = [emitter_idx] * self.params.emitter_population
    emitters[emitter_idx].pop['born'] = [generation] * self.params.emitter_population
    emitters[emitter_idx].pop['evaluated'] = [None] * self.params.emitter_population
    if emitters[emitter_idx].ancestor['ancestor'] is not None:
      parent_ancestor = emitters[emitter_idx].ancestor['ancestor']
    else:
      parent_ancestor = emitter_idx
    emitters[emitter_idx].pop['ancestor'] = [parent_ancestor] * self.params.emitter_population

  def update_cmaes_values(self, emitters, emitter_idx):
    """
    This function passes to the CMAES emitter the values and genomes
    :param emitter_idx: Index of the emitter in the emitter list
    :return:
    """
    # There is a - in front of the value cause the CMA-ES minimizes the values, while I want to maximize them
    solutions = [(genome, -value) for genome, value in zip(emitters[emitter_idx].pop['genome'], emitters[emitter_idx].pop['reward'])]
    emitters[emitter_idx].tell(solutions)

  def candidate_emitter_eval(self, evaluate_in_env, budget_chunk, generation, pool=None):
    """
    This function does a small evaluation for the cadidate emitters to calculate their initial improvement
    :return:
    """
    candidates = self.candidates_by_novelty(pool=pool)
    for candidate in candidates:
      # Bootstrap candidates improvements
      if budget_chunk <= self.params.chunk_size/3 or self.evaluation_budget <= 0:
        break

      rew_area = 'rew_area_{}'.format(self.emitter_candidate[candidate].ancestor['rew_area'])
      if rew_area not in Logger.data:
        Logger.data[rew_area] = 0

      for i in range(6):
        self.update_cmaes_population(self.emitter_candidate, candidate, generation)
        evaluate_in_env(self.emitter_candidate[candidate].pop, pool=pool)
        self.emitter_candidate[candidate].pop['evaluated'] = list(range(self.evaluated_points,
                                                                 self.evaluated_points + self.params.emitter_population))

        self.update_cmaes_values(self.emitter_candidate, candidate)

        self.emitter_candidate[candidate].values.append(self.emitter_candidate[candidate].pop['reward'])
        self.update_reward_archive(generation, self.emitter_candidate, candidate)

        # Update counters
        # step_count += 1
        self.emitter_candidate[candidate].steps += 1
        self.evaluated_points += self.emitter_candidate[candidate].pop.size
        self.evaluation_budget -= self.emitter_candidate[candidate].pop.size
        budget_chunk -= self.emitter_candidate[candidate].pop.size
        Logger.data[rew_area] += self.emitter_candidate[candidate].pop.size

      self.emitter_candidate[candidate].estimate_improvement()

      # Add to emitters list
      if self.emitter_candidate[candidate].improvement > 0:
        self.emitters[candidate] = copy.deepcopy(self.emitter_candidate[candidate])
      del self.emitter_candidate[candidate]
    return budget_chunk

  def emitter_step(self, searcher_params, evaluate_in_env, generation, ns_pop, ns_off, budget_chunk, pool=None):
    """
    This function performs the steps for the CMA-ES emitters
    :param evaluate_in_env: Function used to evaluate the agents in the environment
    :param generation: Generation at which the process is
    :param ns_pop: novelty search population
    :param ns_off: novelty search offsprings
    :param budget_chunk: budget chunk to allocate to search
    :param pool: Multiprocessing pool
    :return:
    """

    budget_chunk = self.candidate_emitter_eval(evaluate_in_env, budget_chunk, generation, pool)

    ns_reference_set = self.get_novelty_ref_set(ns_pop, ns_off)

    while self.emitters and budget_chunk > 0 and self.evaluation_budget > 0: # Till we have emitters or computation budget
      emitter_idx = self.choose_emitter()

      # Calculate parent novelty
      self.emitters[emitter_idx].ancestor['novelty'] = utils.calculate_novelties(
        [self.emitters[emitter_idx].ancestor['bd']],
        ns_reference_set,
        distance_metric=consts.serene_novelty_distance_metric,
        novelty_neighs=consts.serene_k_novelty_neighs,
        pool=pool
      )[0]


      # Only store most novel if never savedS
      # if self.emitters[emitter_idx].most_novel is None:
      #   self.emitters[emitter_idx].most_novel = self.rewarding[emitter_idx].copy()
      # self.rew_archive.store(self.rewarding[emitter_idx])  # Always store parent

      # step_count = 0
      print("Emitter: {} - Improv: {}".format(emitter_idx, self.emitters[emitter_idx].improvement))
      rew_area = 'rew_area_{}'.format(self.emitters[emitter_idx].ancestor['rew_area'])

      # The emitter evaluation cycle breaks every X steps to choose a new emitter
      # ---------------------------------------------------
      while budget_chunk > 0 and self.evaluation_budget > 0:
        self.update_cmaes_population(self.emitters, emitter_idx, generation)
        evaluate_in_env(self.emitters[emitter_idx].pop, pool=pool)
        self.emitters[emitter_idx].pop['evaluated'] = list(range(self.evaluated_points,
                                                                 self.evaluated_points + self.params.emitter_population))

        self.update_cmaes_values(self.emitters, emitter_idx)

        self.emitters[emitter_idx].values.append(self.emitters[emitter_idx].pop['reward'])
        self.update_reward_archive(generation, self.emitters, emitter_idx)

        # Now calculate novelties and update most novel
        self.update_emitter_novelties(ns_ref_set=ns_reference_set, ns_pop=ns_pop, emitter_idx=emitter_idx, pool=pool)

        # Update counters
        # step_count += 1
        self.emitters[emitter_idx].steps += 1
        self.evaluated_points += self.emitters[emitter_idx].pop.size
        self.evaluation_budget -= self.emitters[emitter_idx].pop.size
        budget_chunk -= self.emitters[emitter_idx].pop.size
        Logger.data[rew_area] += self.emitters[emitter_idx].pop.size

        if self.check_stopping_criteria(emitter_idx):
          self.emitters_data[int(emitter_idx)] = {'generation': generation,
                                                  'steps': self.emitters[emitter_idx].steps,
                                                  'rewards': self.emitters[emitter_idx].values,
                                                  'archived': self.emitters[emitter_idx].archived}

          self.archive_candidates[emitter_idx] = copy.deepcopy(self.emitters[emitter_idx].ns_arch_candidates)
          # Always store parent once emitter is dead so we do not add it many times
          self.rew_archive.store(self.emitters[emitter_idx].ancestor)
          print("Stopped after {} steps\n".format(self.emitters[emitter_idx].steps))
          del self.emitters[emitter_idx]
          break
      # ---------------------------------------------------
      # This is done only if the emitter still exists
      if emitter_idx in self.emitters:
        self.emitters[emitter_idx].estimate_improvement()
