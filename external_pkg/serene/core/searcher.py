# Created by Giuseppe Paolo 
# Date: 27/07/2020

#Here I make the class that creates everything. I pass the parameters as init arguments, this one creates the param class, and the pop, arch, opt alg

import os
import pdb

from external_pkg.serene.core.population import Population
from external_pkg.serene.core.evolvers import NoveltySearch, CMAES, CMANS, NSGAII, SERENE, MAPElites, CMAME, RandomSearch
from external_pkg.serene.core.serene_utils import evaluate_grasp_serene
from external_pkg.serene.core.behavior_descriptors.behavior_descriptors import BehaviorDescriptor
from external_pkg.serene.core import Evaluator
import multiprocessing as mp
#from timeit import default_timer as timer
import gc
from external_pkg.serene.analysis.logger import Logger

from algorithms.population import Population

from utils.io_run_data import dump_archive_success_routine
from utils.evo_tools import replace_pop
from utils.evo_main_routines import select_offspring_routine, mutate_offspring_routine
from utils.novelty_computation import update_novelty_routine
import utils.constants as consts

import numpy as np

evaluator = None
main_pool = None # Using pool as global prevents the creation of new environments at every generation




class Searcher(object):
  """
  This class creates the instance of the NS algorithm and everything related
  """
  def __init__(self, parameters, id_counter):
    self.parameters = parameters

    '''
    self.bd_extractor = BehaviorDescriptor(self.parameters)
    '''

    self.generation = 1

    ''' handeled with scoop
    if self.parameters.multiprocesses:
      global main_pool
      main_pool = mp.Pool(initializer=self.init_process, processes=self.parameters.multiprocesses)
    
    self.evaluator = Evaluator(self.parameters)
    
    self.population = Population(self.parameters, init_size=self.parameters.pop_size)
    '''

    self.population = Population(
        toolbox=parameters['toolbox'],
        max_pop_size=parameters['pop_size'],
        genotype_len=parameters['genotype_len'],
        n_reinit_flg=parameters['reinit_research_flg'],
        bound_genotype_thresh=parameters['bound_genotype_thresh'],
        closer_genome_init_func=parameters['closer_genome_init_func'],
        curr_n_evals=0,
        prob_cx=parameters['prob_cx'],  #  should be useless here : to refactore
        id_counter=id_counter
      )
    self.init_pop = True
    self.offsprings = None

    self.evolver = SERENE(self.parameters)
  '''
  def init_process(self):
    """
    This function is used to initialize the pool so each process has its own instance of the evaluator
    :return:
    """
    global evaluator
    evaluator = Evaluator(self.parameters)
  '''
  '''
  def _feed_eval(self, agent):
    """
    This function feeds the agent to the evaluator and returns the updated agent
    :param agent:
    :return:
    """
    global evaluator
    if agent['evaluated'] == None: # Agents are evaluated only once
      agent = evaluator(agent, self.bd_extractor.__call__)
    return agent
  '''

  def evaluate_in_env(self, pop, pool=None):
    """
    This function evaluates the population in the environment by passing it to the parallel evaluators.
    :return:
    """
    if self.parameters.verbose: print('Evaluating {} in environment.'.format(pop.name))
    if pool is not None:
      pop.pop = pool.map(self._feed_eval, pop.pop) # As long as the ID is fine, the order of the element in the list does not matter
    else:
      for i in range(pop.size):
        if self.parameters.verbose: print(".", end = '') # The end prevents the newline
        if pop[i]['evaluated'] is None: # Agents are evaluated only once
          pop[i] = self.evaluator(pop[i], self.bd_extractor)
      if self.parameters.verbose: print()

  def _main_search(self, qds_args, budget_chunk, searcher_params, id_counter):
    """
    This function performs the main search e.g. NS/NSGA/CMA-ES
    :return:
    """
    toolbox = qds_args['toolbox']
    evaluator = qds_args['evaluator']
    progression_monitoring = qds_args['progression_monitoring']
    outcome_archive = qds_args['outcome_archive']  # todo bien mise à jour ? ou copie crée ?
    archive = qds_args['archive']
    stats_tracker = qds_args['stats_tracker']  # todo bien mise à jour ? ou copie crée ?
    prob_cx = qds_args['prob_cx']

    fixed_attr_dict = {
        'toolbox': toolbox,
        'pop_size': qds_args['pop_size'],
        'genotype_len': qds_args['genotype_len'],
        'reinit_research_flg': qds_args['reinit_research_flg'],
        'bound_genotype_thresh': qds_args['bound_genotype_thresh'],
        'closer_genome_init_func': qds_args['closer_genome_init_func'],
        'algo_variant': qds_args['algo_variant'],
        'outcome_archive_kwargs': qds_args['outcome_archive_kwargs'],
        'evaluator': evaluator,
        'is_novelty_required': qds_args['is_novelty_required'],
        'evo_process': qds_args['evo_process'],
        'bd_indexes': qds_args['bd_indexes'],
        'bd_filters': qds_args['bd_filters'],
        'novelty_metric': qds_args['novelty_metric'],
        'bd_bounds': qds_args['bd_bounds'],
        'nb_offsprings_to_generate': qds_args['nb_offsprings_to_generate'],
        'mut_flg': qds_args['mut_flg'],
        'prob_cx': qds_args['prob_cx'],
        'robot_name': qds_args['robot_name'],
        'obj_vertices_poses': qds_args['obj_vertices_poses'],
        'stabilized_obj_pose': qds_args['stabilized_obj_pose'],
    }

    # Evaluate population in the environment only the first time
    if self.init_pop:
      objective_batch, measure_batch, infos_batch = evaluate_grasp_serene(
        toolbox=toolbox, evaluator=evaluator, inds=self.population.inds
      )

      self.population.update_individuals(
        fitnesses=[(fit,) for fit in objective_batch],
        b_descriptors=measure_batch,
        infos=infos_batch,
        curr_n_evals=progression_monitoring.n_eval
      )

      novelty_pop = update_novelty_routine(
          is_novelty_required=qds_args['is_novelty_required'],
          inds2update=self.population.inds,
          archive=archive,
          evo_process=qds_args['evo_process'],
          bd_indexes=qds_args['bd_indexes'],
          bd_filters=qds_args['bd_filters'],
          novelty_metric=qds_args['novelty_metric'],
          algo_variant=qds_args['algo_variant'],
          bd_bounds=qds_args['bd_bounds']
      )

      archive.fill(
          pop2add=self.population.inds,
          novelties=novelty_pop
      )

      added_inds, scs_inds_generated = outcome_archive.update(self.population)
      progression_monitoring.update(pop=self.population, outcome_archive=outcome_archive)
      stats_tracker.update(
        pop=self.population, outcome_archive=outcome_archive,
        curr_n_evals=progression_monitoring.n_eval, gen=self.generation
      )

      self.evolver.evaluated_points += len(self.population)
      self.evolver.evaluation_budget -= len(self.population)
      budget_chunk -= len(self.population)

      #pdb.set_trace()  # ok
      self.init_pop = False

    ref_pop_inds = self.population

    while budget_chunk > 0 and self.evolver.evaluation_budget > 0:

      self.offsprings = select_offspring_routine(
        pop=self.population,
        ref_pop_inds=ref_pop_inds,
        archive=None,
        id_counter=id_counter,
        **fixed_attr_dict
      )

      off_bds = self.offsprings.get_behavior_descriptors()
      #pdb.set_trace()
      n_touch = np.sum([1 if bd != [0., 0., 0.] else 0 for bd in off_bds])
      #print('NS off_bds=', off_bds)
      #print('NS off n_touch=', n_touch)

      mutate_offspring_routine(
        off=self.offsprings,
        outcome_archive=outcome_archive,
        pop=self.population,
        gen=self.generation,
        **fixed_attr_dict
      )
      id_counter = self.offsprings.id_counter

      ref_pop_inds = self.population.inds + self.offsprings.inds
      invalid_inds = self.population.get_invalid_inds() + self.offsprings.get_invalid_inds()

      inv_pop = Population(
        inds=invalid_inds, toolbox=toolbox, genotype_len=qds_args['genotype_len'],
        id_counter=id_counter, prob_cx=prob_cx
      )

      objective_batch, measure_batch, infos_batch = evaluate_grasp_serene(
          toolbox=toolbox, evaluator=evaluator, inds=inv_pop.inds
      )

      inv_pop.update_individuals(
          fitnesses=[(fit,) for fit in objective_batch],
          b_descriptors=measure_batch,
          infos=infos_batch,
          curr_n_evals=progression_monitoring.n_eval
      )


      novelty_updated_inds = update_novelty_routine(
        inds2update=ref_pop_inds,
        archive=archive,
        **fixed_attr_dict
      )

      added_inds, scs_inds_generated = outcome_archive.update(inv_pop)
      progression_monitoring.update(pop=inv_pop, outcome_archive=outcome_archive)

      stats_tracker.update(
        pop=inv_pop, outcome_archive=outcome_archive,
        curr_n_evals=progression_monitoring.n_eval, gen=self.generation
      )

      self.evolver.evaluated_points += len(inv_pop)
      self.evolver.evaluation_budget -= len(inv_pop)
      budget_chunk -= len(inv_pop)

      if inv_pop.get_n_successful_inds() > 0:
        n_gen_rolling_reinit_research = 0

      id_counter = self.evolver.init_emitters(parameters=self.parameters, ns_pop=self.population, ns_off=self.offsprings,
                                 id_counter=id_counter)
      #print('after emitter : id_counter=', id_counter) # ok
      #pdb.set_trace()

      # ---------------------------------------- FILL ARCHIVE --------------------------------------

      novelties_off = novelty_updated_inds[qds_args['pop_size']:] \
          if qds_args['algo_variant'] in consts.POP_BASED_ALGO_VARIANTS else novelty_updated_inds

      archive.fill(
          pop2add=self.offsprings.inds,
          novelties=novelties_off
      )

      # ------------------------------------------ REPLACE -----------------------------------------
      replace_pop(
          pop=self.population,
          ref_pop_inds=ref_pop_inds,
          **fixed_attr_dict
      )

      # -------------------------------------- MANAGE ARCHIVE --------------------------------------

      archive.manage_archive_size()

      # ----------------------------------------- MEASURE ------------------------------------------


      do_dump_scs_archive = consts.DUMP_SCS_ARCHIVE_ON_THE_FLY and self.generation % consts.N_GEN_FREQ_DUMP_SCS_ARCHIVE == 0
      if do_dump_scs_archive:
          dump_archive_success_routine(
              timer=qds_args['timer'],
              timer_label=consts.NS_RUN_TIME_LABEL,
              run_name=qds_args['run_name'],
              curr_neval=progression_monitoring.n_eval,
              outcome_archive=outcome_archive,
          )

      self.generation += 1

    return id_counter

  def _emitter_search(self, qds_args, searcher_params, budget_chunk, id_counter):
    """
    This function performs the reward search through the emitters
    :return:
    """

    if self.evolver.emitter_based and (len(self.evolver.emitters) > 0 or len(self.evolver.emitter_candidate) > 0):

      self.evolver.emitter_step(
          searcher_params=searcher_params,
          qds_args=qds_args,
          evaluate_in_env=self.evaluate_in_env,
          generation=self.generation,
          ns_pop=self.population,
          ns_off=self.offsprings,
          budget_chunk=budget_chunk,
          pool=None,
          id_counter=id_counter
      )
      #self.evolver.rew_archive.save(self.parameters.save_path, 'gen_{}'.format(self.generation))

      # todo : eval update, and "elaborate" ?

      # Update the performaces due to possible changes in the pop and archive given by the emitters
      self.evolver.evaluate_performances(self.population, self.offsprings, pool=None)
      # Update main archive with the archive candidates from the emitters
      self.evolver.elaborate_archive_candidates(parameters=searcher_params, generation=self.generation)

    return id_counter

  def chunk_step(self, qds_args, searcher_params, id_counter):
    """
    This function performs all the calculations needed for one generation.
    Generates offsprings, evaluates them and the parents in the environment, calculates the performance metrics,
    updates archive and population and finally saves offsprings, population and archive.
    :return: time taken for running the generation
    """
    #global main_pool
    #start_time = timer()

    print("\nRemaining budget: {}".format(self.evolver.evaluation_budget))

    # -------------------
    # Base part
    # -------------------
    budget_chunk = consts.serene_chunk_size
    if self.evolver.evaluation_budget > 0:
      print("MAIN")
      n_scs_before_ms = len(qds_args['outcome_archive'].get_successful_inds())
      print('before main search : n_scs_before_ms = ', n_scs_before_ms)
      id_counter = self._main_search(qds_args=qds_args, searcher_params=searcher_params, budget_chunk=budget_chunk,
                        id_counter=id_counter)
      n_scs_after_ms = len(qds_args['outcome_archive'].get_successful_inds())
      print('after main search : n_scs_after_ms = ', n_scs_after_ms)


    #pdb.set_trace()

    # -------------------
    # Emitters part
    # -------------------
    budget_chunk = consts.serene_chunk_size
    # Starts only if a reward has been found.
    if self.evolver.evaluation_budget > 0:
      print("EMITTERS: {}".format(len(self.evolver.emitters))) # TODO REPRENDRE ICI (en forçant une fit, on devrait retourner dans l'ajout des emit là haut)

      n_scs_before_es = len(qds_args['outcome_archive'].get_successful_inds())
      print('before emitter search : len(archive_scs) = ', n_scs_before_es)

      id_counter = self._emitter_search(qds_args=qds_args, searcher_params=searcher_params, budget_chunk=budget_chunk,
                           id_counter=id_counter)

      n_scs_after_es = len(qds_args['outcome_archive'].get_successful_inds())
      print('after emitter search : len(archive_scs) = ', n_scs_after_es)
    # -------------------

    return id_counter #self.evolver.evaluated_points
