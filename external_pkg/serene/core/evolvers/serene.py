# Created by giuseppe
# Date: 05/10/20
import pdb

import numpy as np
import copy

from external_pkg.serene.core.population import Population as SerenePopulation
from external_pkg.serene.core.evolvers import EmitterEvolver, utils
from external_pkg.serene.core.serene_utils import evaluate_grasp_serene

from algorithms.population import Population

import utils.constants as consts


class FitnessEmitter(object):
  """
  This class implements the fitness emitter
  """
  def __init__(self, ancestor, mutation_rate, parameters, id_counter):

    self.ancestor = ancestor
    self._init_mean = np.array(self.ancestor).tolist()
    self.id = ancestor.gen_info.values['id']
    self._mutation_rate = mutation_rate
    self._pop_size = consts.emitter_population_len

    self.pop = self._init_pop(
      toolbox=parameters['toolbox'], id_counter=id_counter, prob_cx=parameters['prob_cx'],
      curr_n_evals=parameters['progression_monitoring'].n_eval, parameters=parameters
    )
    self.ns_arch_candidates = SerenePopulation(parameters, init_size=0, name='ns_arch_cand')

    # List of lists. Each inner list corresponds to the values obtained during a step
    # We init with the ancestor reward so it's easier to calculate the improvement
    self.values = []
    self.archived_values = []
    self.improvement = 0
    self._init_values = None
    self.archived = []  # List containing the number of archived agents at each step
    self.most_novel = self.ancestor
    self.steps = 0

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
    if consts.serene_emitter_local_improvement_flg:
      self._init_values = self.values[-3:]


  def mutate_genome(self, parameters, genome):
    """
    This function mutates the genome
    :param genome:
    :return:
    """
    genome = genome + np.random.normal(0, self._mutation_rate, size=np.shape(genome))
    return genome.clip(parameters['genome_limit'][0], parameters['genome_limit'][1])

  def _init_pop(self, toolbox, id_counter, prob_cx, curr_n_evals, parameters):
    """
    This function initializes the emitter pop around the parent
    :return:
    """
    mutated_genomes = [self.mutate_genome(genome=self._init_mean, parameters=parameters) for _ in range(self._pop_size)]

    # init deap pop
    pop = Population(
      toolbox=toolbox, id_counter=id_counter, prob_cx=prob_cx, cx_flg=True, curr_n_evals=curr_n_evals,
      max_pop_size=self._pop_size, len_pop=self._pop_size
    )
    # set genomes to each individuals
    pop.set_genomes_to_deap_pop(genomes=mutated_genomes)
    return pop

  def generate_off(self, parameters, generation, toolbox, id_counter, prob_cx, curr_n_evals):
    """
    This function generates the offsprings of the emitter
    :return:
    """

    # init deap pop
    off = Population(
      toolbox=toolbox, id_counter=id_counter, prob_cx=prob_cx, cx_flg=True, curr_n_evals=curr_n_evals,
      max_pop_size=2*self._pop_size, len_pop=2*self._pop_size
    )

    # generate 2 offspring from each parent
    mutated_genomes = []
    for agent in self.pop:
      mutated_genomes.append(self.mutate_genome(parameters=parameters, genome=agent))
      mutated_genomes.append(self.mutate_genome(parameters=parameters, genome=agent))

    # set genomes to each individuals
    off.set_genomes_to_deap_pop(genomes=mutated_genomes)

    return off

  def update_pop(self, offsprings):
    """
    This function chooses the agents between the pop and the off with highest reward to create the new pop
    :param offsprings:
    :return:
    """

    performances = self.pop.get_fitnesses() + offsprings.get_fitnesses()

    idx = np.argsort(performances)[::-1]  # Order idx according to performances.
    parents_off = self.pop.inds + offsprings.inds

    # Update population list by going through it and putting an agent from parents+off at its place
    for new_pop_idx, old_pop_idx in zip(range(len(self.pop)), idx[:len(self.pop)]):
      assert len(self.pop.inds[new_pop_idx]) == len(parents_off[old_pop_idx])
      self.pop.inds[new_pop_idx] = parents_off[old_pop_idx]

    return

  def should_stop(self):
    """
    Checks internal stopping criteria
    :return:
    """
    return False


class SERENE(EmitterEvolver):
  """
  This class implements the SERENE evolver. It performs NS till a reward is found, then launches fitness based emitters
  to search the reward area.
  """
  def create_emitter(self, parameters, parent_id, ns_pop, ns_off, id_counter):
    """
    This function creates the emitter
    :param parent_id:
    :param ns_pop:
    :param ns_off:
    :return:
    """

    return FitnessEmitter(
      ancestor=self.rewarding[parent_id],
      mutation_rate=self.calculate_init_sigma(
        parameters=parameters, ns_pop=ns_pop, ns_off=ns_off, mean=self.rewarding[parent_id]
      ),
      parameters=parameters,
      id_counter=id_counter
    )

  def candidate_emitter_eval(self, searcher_params, qds_args, evaluate_in_env, budget_chunk, generation, id_counter,
                             pool=None):
    """
    This function does a small evaluation for the cadidate emitters to calculate their initial improvement
    :return:
    """

    n_scs_before_emit = len(qds_args['outcome_archive'].get_successful_inds())
    print(f'(candidate_emitter_eval) n_scs_before_emit={n_scs_before_emit}')

    candidates = self.candidates_by_novelty(parameters=searcher_params, pool=pool)

    toolbox = qds_args['toolbox']
    progression_monitoring = qds_args['progression_monitoring']

    for candidate in candidates:
      # Bootstrap candidates improvements
      if budget_chunk <= consts.serene_chunk_size/3 or self.evaluation_budget <= 0:
        break

      # Initial population evaluation
      #-----
      emitter_candidate_subpop = self.emitter_candidate[candidate].pop.inds
      objective_batch, measure_batch, infos_batch = evaluate_grasp_serene(
        toolbox=toolbox, evaluate_fn=qds_args['evaluate_fn'], inds=emitter_candidate_subpop
      )
      self.emitter_candidate[candidate].pop.update_individuals(
        fitnesses=[(fit,) for fit in objective_batch],
        b_descriptors=measure_batch,
        infos=infos_batch,
        curr_n_evals=progression_monitoring.n_eval
      )

      added_inds, scs_inds_generated = qds_args['outcome_archive'].update(self.emitter_candidate[candidate].pop)

      qds_args['progression_monitoring'].update(
        pop=self.emitter_candidate[candidate].pop, outcome_archive=qds_args['outcome_archive']
      )
      qds_args['stats_tracker'].update(
        pop=self.emitter_candidate[candidate].pop, outcome_archive=qds_args['outcome_archive'],
        curr_n_evals=progression_monitoring.n_eval, gen=generation
      )

      # Update counters
      self.evaluated_points += len(self.emitter_candidate[candidate].pop)
      self.evaluation_budget -= len(self.emitter_candidate[candidate].pop)
      budget_chunk -= len(self.emitter_candidate[candidate].pop)

      for i in range(5):  # Evaluate emitter on 6 generations

        #----
        # off generation and eval
        offsprings = self.emitter_candidate[candidate].generate_off(
          parameters=searcher_params,
          generation=generation,
          toolbox=searcher_params['toolbox'],
          id_counter=id_counter,
          prob_cx=searcher_params['prob_cx'],
          curr_n_evals=searcher_params['progression_monitoring'].n_eval,
        )
        id_counter = offsprings.id_counter

        objective_batch, measure_batch, infos_batch = evaluate_grasp_serene(
          toolbox=toolbox, evaluate_fn=qds_args['evaluate_fn'], inds=offsprings.inds
        )
        offsprings.update_individuals(
          fitnesses=[(fit,) for fit in objective_batch],
          b_descriptors=measure_batch,
          infos=infos_batch,
          curr_n_evals=progression_monitoring.n_eval
        )

        self.emitter_candidate[candidate].update_pop(offsprings)
        self.emitter_candidate[candidate].values.append(self.emitter_candidate[candidate].pop.get_fitnesses())

        added_inds, scs_inds_generated = qds_args['outcome_archive'].update(offsprings)
        qds_args['progression_monitoring'].update(
          pop=offsprings, outcome_archive=qds_args['outcome_archive']
        )
        qds_args['stats_tracker'].update(
          pop=offsprings, outcome_archive=qds_args['outcome_archive'],
          curr_n_evals=progression_monitoring.n_eval, gen=generation
        )

        # Update counters
        self.emitter_candidate[candidate].steps += 1
        self.evaluated_points += len(offsprings)
        self.evaluation_budget -= len(offsprings)
        budget_chunk -= len(offsprings)

      self.emitter_candidate[candidate].estimate_improvement()
      print(f'improvement estimation complete (value={self.emitter_candidate[candidate].improvement})')

      # Add to emitters list
      print('self.emitter_candidate[candidate].improvement=', self.emitter_candidate[candidate].improvement)
      if self.emitter_candidate[candidate].improvement > 0:
        self.emitters[candidate] = copy.deepcopy(self.emitter_candidate[candidate])
      else:
        print('discarded emitter: no improvement')

      del self.emitter_candidate[candidate]

    n_scs_after_emit = len(qds_args['outcome_archive'].get_successful_inds())
    print(f'(candidate_emitter_eval) n_scs_after_emit={n_scs_after_emit}')

    return budget_chunk, id_counter

  def emitter_step(
          self, searcher_params, qds_args, evaluate_in_env, generation, ns_pop, ns_off, budget_chunk, id_counter,
          pool=None
  ):
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
    n_scs_before_emit = len(qds_args['outcome_archive'].get_successful_inds())
    print(f'(emitter_step) n_scs_before_emit={n_scs_before_emit}')
    budget_chunk, id_counter = self.candidate_emitter_eval(
      searcher_params=searcher_params,
      qds_args=qds_args,
      evaluate_in_env=evaluate_in_env,
      budget_chunk=budget_chunk,
      generation=generation,
      pool=pool,
      id_counter=id_counter,
    )
    print('candidate_emitter_eval done.')
    print(f'candidate_emitter_eval done: id_counter={id_counter}')
    n_scs_after_emit = len(qds_args['outcome_archive'].get_successful_inds())
    print(f'(emitter_step) n_scs_after_emit={n_scs_after_emit}')


    ns_reference_set = self.get_novelty_ref_set(ns_pop, ns_off)

    while self.emitters and budget_chunk > 0 and self.evaluation_budget > 0:  # Till we have emitters or computation budget
      emitter_idx = self.choose_emitter(parameters=searcher_params)

      # Calculate parent novelty
      parent_nov = utils.calculate_novelties(
          [self.emitters[emitter_idx].ancestor.behavior_descriptor.values],
          ns_reference_set,
          distance_metric=consts.serene_novelty_distance_metric,
          novelty_neighs=consts.serene_k_novelty_neighs,
          pool=pool
        )[0]

      self.emitters[emitter_idx].ancestor.novelty.values = (parent_nov,)

      print("Emitter: {} - Improv: {}".format(emitter_idx, self.emitters[emitter_idx].improvement))

      # The emitter evaluation cycle breaks every X steps to choose a new emitter
      # ---------------------------------------------------
      while budget_chunk > 0 and self.evaluation_budget > 0:
        offsprings = self.emitters[emitter_idx].generate_off(
          parameters=searcher_params,
          generation=generation,
          toolbox=searcher_params['toolbox'],
          id_counter=id_counter,
          prob_cx=searcher_params['prob_cx'],
          curr_n_evals=searcher_params['progression_monitoring'].n_eval
          )

        id_counter = offsprings.id_counter

        #-----
        # evals
        objective_batch, measure_batch, infos_batch = evaluate_grasp_serene(
          toolbox=searcher_params['toolbox'], evaluate_fn=qds_args['evaluate_fn'], inds=offsprings.inds
        )
        offsprings.update_individuals(
          fitnesses=[(fit,) for fit in objective_batch],
          b_descriptors=measure_batch,
          infos=infos_batch,
          curr_n_evals=searcher_params['progression_monitoring'].n_eval
        )

        self.emitters[emitter_idx].update_pop(offsprings)
        self.emitters[emitter_idx].values.append(self.emitters[emitter_idx].pop.get_fitnesses())

        # Now calculate novelties and update most novel
        id_counter = self.update_emitter_novelties(
          parameters=searcher_params, ns_ref_set=ns_reference_set, ns_pop=ns_pop, emitter_idx=emitter_idx, pool=pool,
          id_counter=id_counter
        )

        added_inds, scs_inds_generated = qds_args['outcome_archive'].update(offsprings)
        qds_args['progression_monitoring'].update(
          pop=offsprings, outcome_archive=qds_args['outcome_archive']
        )
        qds_args['stats_tracker'].update(
          pop=offsprings, outcome_archive=qds_args['outcome_archive'],
          curr_n_evals=qds_args['progression_monitoring'].n_eval, gen=generation
        )

        # Update counters
        self.emitters[emitter_idx].steps += 1
        self.evaluated_points += len(offsprings)
        self.evaluation_budget -= len(offsprings)
        budget_chunk -= len(offsprings)

        if self.check_stopping_criteria(parameters=searcher_params, emitter_idx=emitter_idx): # Only if emitter is finished
          self.emitters_data[int(emitter_idx)] = {'generation': generation,
                                                  'steps': self.emitters[emitter_idx].steps,
                                                  'rewards': self.emitters[emitter_idx].values,
                                                  'archived': self.emitters[emitter_idx].archived}

          self.archive_candidates[emitter_idx] = copy.deepcopy(self.emitters[emitter_idx].ns_arch_candidates)

          # Store parent once the emitter is finished
          self.rew_archive.store(self.emitters[emitter_idx].ancestor)
          print("Stopped after {} steps\n".format(self.emitters[emitter_idx].steps))
          del self.emitters[emitter_idx]
          break
        # ---------------------------------------------------
      # This is done only if the emitter still exists
      if emitter_idx in self.emitters:
        self.emitters[emitter_idx].estimate_improvement()

    n_scs_after_emit = len(qds_args['outcome_archive'].get_successful_inds())
    print(f'(emitter_step) n_scs_after_emit={n_scs_after_emit}')



