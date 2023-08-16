# Created by giuseppe
# Date: 19/10/20
import pdb

import numpy as np
import copy
from itertools import chain

from external_pkg.serene.core.evolvers import BaseEvolver
from external_pkg.serene.core.evolvers import utils
from external_pkg.serene.core.population import Archive

import utils.constants as consts


class EmitterEvolver(BaseEvolver):
  """
  This class is the base for the evolver built around the emitter concept
  """
  def __init__(self, parameters, **kwargs):
    super().__init__(parameters)

    self.update_criteria = 'novelty'
    self.rew_archive = Archive(parameters, name='rew_archive')

    self.genome_size = parameters['genome_size']
    self.bounds = parameters['genome_limit'] * np.ones((self.genome_size, len(parameters['genome_limit'])))
    self.emitter_pop = None
    self.emitter_based = True
    self.archive_candidates = {}
    self.emitters = {}
    self.emitter_candidate = {}

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
    population_bd = population.get_behavior_descriptors()
    offsprings_bd = offsprings.get_behavior_descriptors() if offsprings is not None else []

    reference_set = self.get_novelty_ref_set(population, offsprings)
    bd_set = population_bd + offsprings_bd

    novelties = utils.calculate_novelties(bd_set, reference_set, distance_metric=consts.serene_novelty_distance_metric,
                                          novelty_neighs=consts.serene_k_novelty_neighs, pool=pool)
    # Update population and offsprings
    novelties = [(nov,) for nov in novelties]
    population.update_novelties(novelties=novelties[:len(population)])
    offsprings.update_novelties(novelties=novelties[len(population):])
    print("update nov over")


  def calculate_init_sigma(self, parameters, ns_pop, ns_off, mean):
    """
    This function calculates the initial step size value for the emitter
    :param ns_pop:
    :param ns_off:
    :param mean
    :return: step size
    """
    # This is the factor multiplying the std deviation that is used to choose which percentage of the gaussian to select
    number_of_std = 3

    idxs = ns_pop.get_unique_ids() + ns_off.get_unique_ids() if ns_off is not None else ns_pop.get_unique_ids()
    idx = idxs.index(mean.gen_info.values['id'])
    genomes = np.array(ns_pop.inds).tolist() + np.array(ns_off.inds).tolist() if ns_off is not None \
      else np.array(ns_pop.inds).tolist()
    genomes.pop(idx)  # Remove element taken into account

    distances = utils.calculate_distances([mean], np.array(genomes)).flatten()
    sigma = min(distances) / number_of_std
    if sigma == 0.: sigma = parameters['mutation_parameters']['sigma']

    return sigma

  def create_emitter(self, parameters, parent_id, ns_pop, ns_off, id_counter):
    """
    This function creates the emitter
    :param parent:
    :param ns_pop:
    :param ns_off:
    :return:
    """
    raise NotImplementedError

  def init_emitters(self, parameters, ns_pop, ns_off, id_counter):
    """
    This function initializes the emitters.
    :param ns_pop:
    :param ns_off:
    :return: True if any emitter has been init, False otherwise
    """

    n_scs_pop = np.sum([int(ind.info.values['is_success']) for ind in ns_pop.inds])
    n_scs_off = np.sum([int(ind.info.values['is_success']) for ind in ns_off.inds]) if ns_off is not None else 0

    if n_scs_pop > 0 or n_scs_off > 0:
      # Get rewarding genomes
      self.rewarding = {}
      iter_agent = chain(ns_pop, ns_off) if ns_off is not None else ns_pop
      for agent in iter_agent:
        is_non_null_fitness = agent.fitness.values[0] > 0
        is_not_already_emitter = 'emitter' not in agent.info.values
        if is_non_null_fitness and is_not_already_emitter:
          agent.info.values['emitter'] = True
          agent_uid = agent.gen_info.values['id']
          self.rewarding[agent_uid] = agent

          self.rew_archive.store(agent)  # They are also stored in the rew_archive

      # New emitters are added in the candidates list.
      # They will be added to the emitters list only if they can give some improvement
      for rew_agent in self.rewarding:
        self.emitter_candidate[rew_agent] = self.create_emitter(
          parameters=parameters, parent_id=rew_agent, ns_pop=ns_pop, ns_off=ns_off, id_counter=id_counter
        )
        # get the last id_counter after having generated emitter's subpop
        id_counter = self.emitter_candidate[rew_agent].pop.id_counter

      print('len(self.emitter_candidate)=', len(self.emitter_candidate))
      return id_counter

    return id_counter


  def update_reward_archive(self, generation, emitters, emitter_idx):
    """
    This function updates the reward archive. That is the archive in which the rewarding functions found by CMA-ES are
    stored
    :param generation
    :param emitters: List of emitters. Can be either candidate emitters or full emitters
    :param emitter_idx: IDX of evaluated emitter
    :return:
    """
    emitters[emitter_idx].archived.append(0)
    if len(emitters[emitter_idx].archived_values) == 0: # If none added yet, add the one with highest reward
      agent_idx = np.argmax(emitters[emitter_idx].pop['reward'])
      if emitters[emitter_idx].pop[agent_idx]['reward'] > 0:
        emitters[emitter_idx].archived[-1] += 1
        emitters[emitter_idx].pop[agent_idx]['stored'] = generation
        emitters[emitter_idx].archived_values.append(emitters[emitter_idx].pop[agent_idx]['reward'])
        self.rew_archive.store(emitters[emitter_idx].pop[agent_idx])

    else: # Add only the ones whose reward is higher than the max of the emitter
      limit = np.max(emitters[emitter_idx].archived_values)
      for agent_idx in range(emitters[emitter_idx].pop.size):
        if emitters[emitter_idx].pop[agent_idx]['reward'] > limit:
          emitters[emitter_idx].archived[-1] += 1
          emitters[emitter_idx].pop[agent_idx]['stored'] = generation
          emitters[emitter_idx].archived_values.append(emitters[emitter_idx].pop[agent_idx]['reward'])
          self.rew_archive.store(emitters[emitter_idx].pop[agent_idx])

  def check_stopping_criteria(self, parameters, emitter_idx):
    if self.emitters[emitter_idx].steps == consts.serene_max_emitter_steps:
      return True
    elif self.emitters[emitter_idx].should_stop():
      return True
    elif self._stagnation(parameters=parameters, emitter_idx=emitter_idx):
      return True
    else: return False

  def _stagnation(self, parameters, emitter_idx):
    """
    Calculates the stagnation criteria
    :param emitter_idx:
    :param ca_es_step:
    :return:
    """
    if consts.serene_stagnation == 'original':
      bottom = max(int(30 * self.genome_size / consts.emitter_population_len + 120), int(self.emitters[emitter_idx].steps * .2))
      if self.emitters[emitter_idx].steps > bottom:
        limit = int(.3 * bottom)
        values = self.emitters[emitter_idx].values[-bottom:]
        maxes = np.max(values, 1)
        medians = np.median(values, 1)
        if np.median(maxes[:limit]) >= np.median(maxes[-limit:]) and \
                np.median(medians[:limit]) >= np.median(medians[-limit:]):
          return True
      return False
    elif consts.serene_stagnation == 'custom':
      bottom = int(20 * self.genome_size / consts.emitter_population_len + 120)
      if self.emitters[emitter_idx].steps > bottom:
        values = self.emitters[emitter_idx].values[-bottom:]
        if np.median(values[:20]) >= np.median(values[-20:]) or np.max(values[:20]) >= np.max(values[-20:]):
          return True
      return False

  def choose_emitter(self, parameters):
    """
    This function is used to select the emitter with the biggest improvement.
    Emitters are chosen randomnly from the list by weighting the probability by their improvement
    :return:
    """
    em_data_np = []
    for em in self.emitters:
      em_data = [em, self.emitters[em].ancestor.behavior_descriptor.values, self.emitters[em].improvement]
      if not isinstance(em_data[0], int) or not isinstance(em_data[1], list) or not isinstance(em_data[2], float):
        print('Error with em_data (skipping this emitter)=', em_data)
        continue

      em_data_np.append(em_data) # must be homogeneous now

    em_data_np = np.array(em_data_np)
    print('em_data_np=', em_data_np)
    emitters_data = np.atleast_2d(em_data_np)

    if self.rew_archive.size > 0: # Calculate pareto front between Novelty and Improvement
      reference_bd = self.rew_archive['bd']  # + [self.emitters[idx].ancestor['bd'] for idx in self.emitters]
      novelties = utils.calculate_novelties(np.stack(emitters_data[:, 1]),
                                            reference_bd,
                                            distance_metric=consts.serene_novelty_distance_metric,
                                            novelty_neighs=consts.serene_k_novelty_neighs, pool=None)
      fronts = utils.fast_non_dominated_sort(novelties, emitters_data[:, 2]) # Get pareto fronts
      idx = np.random.choice(fronts[0]) # Randomly sample from best front
      return emitters_data[idx, 0] # Return the one on the best front with the highest improv
    else: # Return the one with highest improvement
      idx = np.argmax(emitters_data[:, 2])
      return emitters_data[idx, 0]

  def get_novelty_ref_set(self, ns_pop, ns_off):
    """
    This function extracts the reference set for the novelty calculation
    :param ns_pop:
    :param ns_off:
    :return:
    """
    population_bd = ns_pop.get_behavior_descriptors()
    offsprings_bd = ns_off.get_behavior_descriptors() if ns_off is not None else []

    if self.archive.size > 0:
      archive_bd = self.archive['bd']
    else:
      archive_bd = []
    if self.rew_archive.size > 0:
      rew_archive_bd = self.rew_archive['bd']
    else:
      rew_archive_bd = []
    return population_bd + offsprings_bd + archive_bd + rew_archive_bd

  def update_emitter_novelties(self, parameters, ns_ref_set, ns_pop, emitter_idx, id_counter, pool=None):
    """
    This function updates the most novel agent found by the emitter and the NOVELTY_CANDIDATES_BUFFER.
    It does this by calculating the novelty of the current pop of the emitter.
    :param ns_ref_set: Reference set to calculate Novelty
    :param ns_pop: Novelty Search population
    :param emitter_idx:
    :param pool:
    :return:
    """

    novelties = utils.calculate_novelties(self.emitters[emitter_idx].pop.get_behavior_descriptors(),
                                          ns_ref_set,
                                          distance_metric=consts.serene_novelty_distance_metric,
                                          novelty_neighs=consts.serene_k_novelty_neighs, pool=pool)

    assert len(novelties) == len(self.emitters[emitter_idx].pop.inds)
    for i_ind, ind in enumerate(self.emitters[emitter_idx].pop.inds):
      self.emitters[emitter_idx].pop.inds[i_ind].novelty.values = (novelties[i_ind],)

    # Save in the NS archive candidates buffer the agents with a novelty higher than the previous most novel
    for emitter_agent in self.emitters[emitter_idx].pop:
      if emitter_agent.novelty.values[0] > self.emitters[emitter_idx].most_novel.novelty.values[0]:
        self.emitters[emitter_idx].ns_arch_candidates.add(copy.deepcopy(emitter_agent))

    # Update emitter most novel
    most_novel = np.argmax(novelties)
    if novelties[most_novel] > self.emitters[emitter_idx].most_novel.novelty.values[0]:
      self.emitters[emitter_idx].pop.inds[most_novel].gen_info.values['id'] = id_counter  # Recognize most novel agent by giving it a valid ID
      self.emitters[emitter_idx].pop.inds[most_novel].gen_info.values['parent id'] = emitter_idx  # The emitter idx is saved as the parent of the most novel
      id_counter += 1

      self.emitters[emitter_idx].most_novel = self.emitters[emitter_idx].pop.inds[most_novel].copy()  # Update most novel

    return id_counter

  def candidates_by_novelty(self, parameters, pool=None):
    """
    This function orders the candidates by their novelty wrt the Rew archive.
    This way most novel emitters are evaluated first helpin in better covering the space of reward
    :return: List of candidates idx ordered by novelty if rew_archive.size > 0, else just list of candidates emitters
    """
    # Get list of idx and of parent bd
    candidates_idx = list(self.emitter_candidate.keys())

    if self.rew_archive.size > 0 and len(candidates_idx) > 0:

      reference_bd = self.rew_archive['bd']

      candidates_bd = [self.emitter_candidate[idx].ancestor.behavior_descriptor.values for idx in candidates_idx]

      novelties = utils.calculate_novelties(candidates_bd,
                                            reference_bd,
                                            distance_metric=consts.serene_novelty_distance_metric,
                                            novelty_neighs=consts.serene_k_novelty_neighs, pool=pool)

      # Order candidates idx based on their novelties
      sorted_zipped_lists = sorted(zip(novelties, candidates_idx), reverse=True)
      candidates_idx = [element for _, element in sorted_zipped_lists]
    return candidates_idx

  def update_archive(self, parameters, population, offsprings, generation):
    """
    Updates the archive according to the strategy and the criteria given.
    :param population:
    :param offsprings:
    :return:
    """
    # Get list of ordered indexes according to selection strategy
    if consts.serene_archive_selection_operator == 'random':
      idx = list(range(offsprings.size))
      np.random.shuffle(idx)
    elif consts.serene_archive_selection_operator == 'best':
      performances = offsprings[self.update_criteria]
      idx = np.argsort(performances)[::-1]  # Order idx according to performances. (From highest to lowest)
    else:
      raise ValueError(
        'Please specify a valid selection operator for the archive. Given {} - Valid: ["random", "best"]'.format(
          consts.serene_archive_selection_operator))
    # Add to archive the first lambda offsprings in the idx list
    for i in idx[:consts.serene_archive_n_agents2add]:
      offsprings[i]['stored'] = generation
      self.archive.store(offsprings[i])

  def elaborate_archive_candidates(self, parameters, generation):
    """
    Chooses which archive candidates from the emitters to add in the ns archive
    :param generation:
    :return:
    """
    for em in self.archive_candidates:
      if consts.serene_archive_selection_operator == 'random':
        idx = list(range(self.archive_candidates[em].size))
        np.random.shuffle(idx)
      elif consts.serene_archive_selection_operator == 'best':
        performances = self.archive_candidates[em][self.update_criteria]
        idx = np.argsort(performances)[::-1]  # Order idx according to performances. (From highest to lowest)
      else:
        raise ValueError(
          'Please specify a valid selection operator for the archive. Given {} - Valid: ["random", "best"]'.format(
            consts.serene_archive_selection_operator))

      for i in idx[:consts.serene_archive_n_agents2add]:
        self.archive_candidates[em][i]['stored'] = generation
        self.archive.store(self.archive_candidates[em][i])

    # In this are only the cands of the completed emitters, so it can be emptied after adding to the archive
    self.archive_candidates = {}

  def emitter_step(self, searcher_params, qds_args, evaluate_in_env, generation, ns_pop, ns_off, budget_chunk,
                   id_counter, pool=None):
    """
    This function performs the steps for the FIT emitters
    :param evaluate_in_env: Function used to evaluate the agents in the environment
    :param generation: Generation at which the process is
    :param pool: Multiprocessing pool
    :return:
    """
    raise NotImplementedError


