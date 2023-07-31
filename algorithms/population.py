import time

import numpy as np
import random

import utils.constants as consts
from utils.evo_tools import select_n_multi_bd_tournsize, get_successul_inds_from_set, get_sigma_gauss_from_ind
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.special import softmax

import pdb
import time
from scoop import futures
import os

# trop de valeur par défaut : à refactore au propre
class Population:
    def __init__(self, toolbox, prob_cx, max_pop_size=None, inds=None, genotype_len=None, n_reinit_flg=None,
                 id_counter=None, bound_genotype_thresh=None, closer_genome_init_func=None, cx_flg=False,
                 curr_n_evals=None, len_pop=None):

        # Definitions

        self._inds = []  # list of individuals (deap-based)
        self._max_pop_size = None  # number of inds in the population
        self._prob_cx = prob_cx

        self._genotype_len = genotype_len  # genome's length
        self._toolbox = toolbox  # deap toolbox for evolutionary algorithms utilities

        self._n_reinitialization = 0 if n_reinit_flg else None  # number of reinitialization
        self._global_ind_id_counter = 0 if id_counter is None else id_counter  # rolling unique id for new inds

        self._tournsize = consts.TOURNSIZE

        self._closer_genome_init_func = closer_genome_init_func

        self._cx_flg = cx_flg  # Trigger cross over : False speeds up computation (no useless random() calls)

        # Initializations

        self._init_max_pop_size(max_pop_size=max_pop_size, inds=inds)

        self._init_inds(
            toolbox=toolbox, inds=inds, bound_genotype_thresh=bound_genotype_thresh, curr_n_evals=curr_n_evals,
            len_pop=len_pop
        )


        self._sanity_check_pop_size()

    def _init_inds(self, toolbox, inds, bound_genotype_thresh, curr_n_evals, len_pop=None):

        if inds is not None:
            self.set_all_individuals(inds)
        else:
            if curr_n_evals is None:
                raise RuntimeError(
                    'curr_n_evals must be given as init arg for Population() to initialize the \'n_eval_generated\' '
                    'field field of inds when creating them from scratch.')
            self._init_inds_from_scratch(toolbox=toolbox, curr_n_evals=curr_n_evals, len_pop=len_pop)

        if bound_genotype_thresh is not None:
            self._bound_inds_genotype(bound_genotype_thresh)

    def set_genomes_to_deap_pop(self, genomes):
        assert len(genomes) == len(self.inds)

        for i_gene, gene in enumerate(genomes):
            deap_type = type(self.inds[i_gene])
            for i_val, val in enumerate(gene):
                self.inds[i_gene][i_val] = val

            assert isinstance(self.inds[i_gene], deap_type)

    def _init_max_pop_size(self, max_pop_size, inds):
        if max_pop_size is None:
            assert inds is not None
        self._max_pop_size = len(inds) if max_pop_size is None else max_pop_size

    def __len__(self):
        # Note : does pop_size change during execution ? do we need to keep it in memory ?
        return len(self._inds)

    def __iter__(self):
        for ind in self._inds:
            yield ind

    @property
    def inds(self):
        return self._inds

    @property
    def toolbox(self):
        return self._toolbox

    @property
    def genotype_len(self):
        return self._genotype_len

    @property
    def n_reinitialization(self):
        return self._n_reinitialization

    @property
    def id_counter(self):
        return self._global_ind_id_counter

    @id_counter.setter
    def id_counter(self, value):
        self._global_ind_id_counter = value

    def is_empty(self):
        return len(self._inds) == 0

    def set_all_inds_from_same_genome_debug(self, genome):
        len_genome = len(genome)
        ind_ids_iterable = range(len(self._inds))
        for i_ind in ind_ids_iterable:
            for i_gene in range(len_genome):
                self._inds[i_ind][i_gene] = genome[i_gene]

    def _init_inds_from_scratch(self, toolbox, curr_n_evals, len_pop=None):
        if len_pop is not None:
            self._inds = toolbox.population(n=len_pop)
        else:
            self._inds = toolbox.population(n=self._max_pop_size)

        if self._closer_genome_init_func is not None:
            ind_ids_iterable = range(len(self._inds))

            closer_genomes = list(futures.map(self._closer_genome_init_func, ind_ids_iterable))
            print('closer_genomes=', closer_genomes)

            #pdb.set_trace()
            assert len(closer_genomes) > 0
            len_genome = len(closer_genomes[0])
            for i_ind, genome in zip(ind_ids_iterable, closer_genomes):
                for i_gene in range(len_genome):
                    self._inds[i_ind][i_gene] = genome[i_gene]

        for ind in self._inds:
            self._init_ind(ind, curr_n_evals=curr_n_evals)

    def set_all_individuals(self, individuals):
        """Take given argument as new individuals. It musts match the pop_size length."""
        if self._max_pop_size is not None:
            assert len(individuals) == self._max_pop_size
        self._inds = individuals

    def _sanity_check_pop_size(self):
        # the point of this check was to assure that given inds length matches len given by the toolbox
        # ... which is not the case with serene, as we have the same pop object that will be instanciated as full pop
        # (for ns) and small subpop (cma_es emitters)
        assert self._max_pop_size == len(self._inds)


    def append_individuals(self, individuals):
        self._inds += individuals

    def clear_individuals(self):
        """Delete all individuals."""
        self._inds = []

    def reinitialize_pop(self, curr_n_evals):
        """Reinitilize the whole population to resart exploration from scratch. Meant to be used to reinitialize
        the explore population as soon as a promising individual had been found and added to the refining population."""

        if curr_n_evals is None:
            raise RuntimeError(
                'curr_n_evals must be given as init arg for reinitialize_pop() to initialize the \'n_eval_generated\' '
                'field field of inds when creating them from scratch.')

        self.clear_individuals()
        self._init_inds_from_scratch(toolbox=self.toolbox, curr_n_evals=curr_n_evals)
        #self._fill_population(force_rand_init=True, curr_n_evals=curr_n_evals)
        self._n_reinitialization += 1

    ''' # WARNING : weird that it is not used /!\ how is it handled ?
    def clear_fitnesses(self):
        """Clear individuals' fitnesses in order to make them ready to be re-evaluated."""
        for ind in self._inds:
            del ind.fitness.values
    '''

    def __add__(self, pop2add):
        res_pop = Population.copy(self)
        res_pop.append_individuals(pop2add.inds)
        return res_pop

    # SLOW METHOD : optimizing it
    #def __copy__(self): # later : proper way to write our own __copy__ (or __deepcopy__ if needed)
    @staticmethod
    def copy(pop):
        new_pop = Population(toolbox=pop.toolbox, genotype_len=pop.genotype_len)
        cloned_inds = list(map(pop.toolbox.clone, pop.inds))
        new_pop.set_all_individuals(cloned_inds)
        return new_pop

    def _bound_inds_genotype(self, bound_genotype_thresh):
        for ind in self._inds:
            ind = list(map(lambda y: max(-bound_genotype_thresh, min(bound_genotype_thresh, y)), ind))

    def _get_new_unique_ind_id(self):
        new_id = self._global_ind_id_counter
        self._global_ind_id_counter += 1
        return new_id



    def _init_ind(self, ind, curr_n_evals=None):
        ind.gen_info.values = dict()
        ind.gen_info.values['id'] = self._get_new_unique_ind_id()
        ind.gen_info.values['parent id'] = -1  # by convention
        ind.gen_info.values['age'] = 0  # by convention
        ind.gen_info.values['n_gen_since_scs_parent'] = None
        ind.gen_info.values['scs_parent_genome'] = None
        ind.gen_info.values['parent_genome'] = None
        ind.gen_info['n_eval_generated'] = curr_n_evals

    def get_behavior_descriptors(self):
        return [ind.behavior_descriptor.values for ind in self.inds]

    def get_unique_ids(self):
        return [ind.gen_info.values['id'] for ind in self.inds]

    def get_fitnesses(self):
        return [ind.fitness.values[0] for ind in self.inds]

    def evaluate_and_update_inds(self, evaluator, curr_n_evals=None):
        b_descriptors, fitnesses, infos = self.evaluate(
            evaluator=evaluator,
        )
        self.update_individuals(fitnesses, b_descriptors, infos, curr_n_evals=curr_n_evals)

    def evaluate(self, evaluator):
        """Evaluate the current population using evaluation function contained within evaluator.

        :param evaluator: Evaluator object that contains the evaluate_individual function.
        :return: b_descriptors : list containing behavior descriptor associated to each individual
        :return: fitnesses : list containing fitness associated to each individual
        :return: infos : list containing dicts for information associated to each individual
                         (keys : 'energy', 'reward', 'is_success', 'end effector position relative object',
                         'end effector xyzw relative object', 'diversity_descriptor')
        """

        eval_func = evaluator.ind_evaluation_func

        evaluation_pop = list(self._toolbox.map(eval_func, self._inds))

        b_descriptors, fitnesses, infos = map(list, zip(*evaluation_pop))

        # b_descriptors :  [ behavior_1, ..., behavior_n_pop ]
        # fitnesses :  [ fitnesses_1, ..., fitnesses_n_pop ]
        # infos :  [ infos_1, ..., infos_n_pop ]

        return b_descriptors, fitnesses, infos

    def update_individuals(self, fitnesses, b_descriptors, infos, curr_n_evals=None):
        """Update individuals with the given fitnesses, bds and infos."""
        for ind, fit, bd, inf in zip(self._inds, fitnesses, b_descriptors, infos):
            ind.behavior_descriptor.values = bd
            ind.info.values = inf
            #print('(debug) fit=', fit)
            ind.fitness.values = fit  # note: custom fitness for NSLC (novelty, local_quality) to use it with selNSGA2

            if curr_n_evals is not None:
                ind.gen_info['n_eval_generated'] = curr_n_evals



    def update_scs2scs_metrics_ind(self, ind):
        assert not ind.fitness.valid
        if not ind.fitness.valid:
            try :
                is_parent_a_success = ind.info.values['is_success']
                #print('-'*10)
                #print('ind.info.values =', ind.info.values)
                #print('pas de prob')
            except:
                is_parent_a_success = False
                print('Warning : ind.info.values issue.')
                #print('self._inds=', self._inds)
                #print('ind =', ind)
                #print('ind.info =', ind.info)
                #print('ind.info.values =', ind.info.values)
                #print('ind.info.values =', ind.info.values)

                #for _ in range(1000000):
                #    a = 2**4 # MANUALLY WAIT
                #time.sleep(60)
                #pdb.set_trace()
                #raise RuntimeError()

            if is_parent_a_success:
                ind.gen_info.values['n_gen_since_scs_parent'] = 0
                ind.gen_info.values['scs_parent_genome'] = ind.gen_info.values['parent_genome']
            else:
                is_descending_from_a_scs_ind = ind.gen_info.values['n_gen_since_scs_parent'] is not None
                if is_descending_from_a_scs_ind:
                    assert isinstance(ind.gen_info.values['n_gen_since_scs_parent'], int)
                    ind.gen_info.values['n_gen_since_scs_parent'] += 1
        return ind

    def update_scs2scs_metrics(self):
        self._inds = [self.update_scs2scs_metrics_ind(ind) for ind in self._inds]
        """
        for ind in off.inds:
            if not ind.fitness.valid:
                is_parent_a_success = ind.info.values['is_success']
                if is_parent_a_success:
                    ind.info.values['n_gen_since_scs_parent'] = 0
                else:
                    is_descending_from_a_scs_ind = ind.info.values['n_gen_since_scs_parent'] is not None
                    if is_descending_from_a_scs_ind:
                        assert isinstance(ind.info.values['n_gen_since_scs_parent'], int)
                        ind.info.values['n_gen_since_scs_parent'] += 1
        """

    def update_novelties(self, novelties):
        """Update individuals with the given novelties."""
        for ind, nov in zip(self._inds, novelties):
            ind.novelty.values = nov

    def update_local_qualities(self, local_qualities):
        """Update individuals with the given local quality."""
        for ind, lq in zip(self._inds, local_qualities):
            ind.info.values['local_quality'] = lq

    @staticmethod
    def update_local_qualities_inds(inds, local_qualities):
        """Update individuals with the given local quality."""
        for ind, lq in zip(inds, local_qualities):
            ind.info.values['local_quality'] = lq

    @staticmethod
    def update_novelties_inds(inds, novelties):  # used to avoid creating Population instance
        """Update given individuals with the given novelties."""
        # note : a single method would be great but compatibility issues with "self"
        for ind, nov in zip(inds, novelties):
            ind.novelty.values = nov

    def get_n_success(self):
        return sum([ind.info.values[consts.SUCCESS_CRITERION] for ind in self._inds])

    def get_invalid_inds(self):
        """Returns a list containing individuals in the pop that have invalid fitness (= new inds)."""
        return [ind for ind in self._inds if not ind.fitness.valid]

    def fill_pop_if_required(self, curr_n_evals):
        is_there_inds2generate = len(self._inds) < self._max_pop_size
        if is_there_inds2generate:
            self._fill_population(curr_n_evals=curr_n_evals)

    def _fill_population(self, curr_n_evals, force_rand_init=False):
        """force_rand_init is ugly, and only due to the REFILL_POP_METHOD constraining mecanism.
        Modify it would make this method more elegant."""

        assert self._max_pop_size is not None
        assert self._max_pop_size != 0

        nb_to_fill = self._max_pop_size - len(self)

        if consts.REFILL_POP_METHOD == 'rand' or force_rand_init:
            new_individuals = self._toolbox.population()[:nb_to_fill]
            for ind in new_individuals:
                self._init_ind(ind, curr_n_evals=curr_n_evals)
            self._inds += new_individuals

        elif consts.REFILL_POP_METHOD == 'copies':
            ref_set_inds = [self._toolbox.clone(ind_ref) for ind_ref in self._inds] # required copy ?
            for _ in range(nb_to_fill):
                ind2copy = random.choice(ref_set_inds)
                new_ind = self._toolbox.clone(ind2copy)
                self._init_ind(new_ind, curr_n_evals=curr_n_evals)
                self._inds.append(new_ind)

        assert len(self._inds) == self._max_pop_size

    def random_sample(self, n_sample, inplace=False):
        """(Selection function) Randomly sample n_sample individuals from the current population."""
        selected_inds = random.sample(self._inds, n_sample)
        if inplace:
            self._inds = selected_inds
        return selected_inds

    def select_tournament_attr(self, n_select, attr, inplace=False):
        """(Selection function) Select n_select individuals based on attr criteria (ex:'fitness', 'novelty')."""
        selected_inds = self._toolbox.select(self._inds, n_select, fit_attr=attr)
        if inplace:
            self._inds = selected_inds
        return selected_inds

    def select_n_multi_bd_tournsize(self, n_select, tournsize, bd_filters, inplace=False):
        """(Selection function) Select n_select individuals based on n_multi_bd_tournsize."""
        selected_inds = select_n_multi_bd_tournsize(self._inds, n_select, tournsize, bd_filters)
        if inplace:
            self._inds = selected_inds
        return selected_inds

    def clone_individuals(self, inplace=False, update_scs2scs_fields=True):
        """Clone individuals and store them as new individuals.
        Usually called in a DEAP way, after having the selection process : the selected individuals in offspring
        population must be cloned before any further transformations."""

        if update_scs2scs_fields:
            #pdb.set_trace()
            #ind.infos.values['is_success']

            if inplace:
                self._inds = [self._toolbox.clone(ind) for ind in self._inds]
                return self._inds
            else:
                return [self._toolbox.clone(ind) for ind in self._inds]

            pass


        if inplace:
            self._inds = [self._toolbox.clone(ind) for ind in self._inds]
            return self._inds
        else:
            return [self._toolbox.clone(ind) for ind in self._inds]

    def var_and(self, bound_genotype_thresh, evo_process, e2r_mode='standard'):
        """Applies variations (crossover and mutation) to the current population.

        Note : the name refers to DEAP's varAnd() function that applies both cx and mutation.

        Args:
            bound_genotype_thresh (float): absolute value bound of genotype values
            evo_process (str) : type of applied algorithm for evolution.
        """

        for id_ind, ind in enumerate(self._inds):

            do_mutation = True
            if self._cx_flg:
                n_inds = len(self._inds)
                is_last_ind = id_ind == (n_inds - 1)  # if issue, replace by '>='
                # apply_crossover_flg = random.random() < consts.CXPB and not is_last_ind
                apply_crossover_flg = random.random() < self._prob_cx and not is_last_ind
                if apply_crossover_flg:
                    # pdb.set_trace()  # debug CX
                    other_parent = self._toolbox.clone(self._inds[id_ind + 1])  # do not modify the next parent then
                    parents_id = [ind.gen_info.values['id'], other_parent.gen_info.values['id']]
                    self._toolbox.mate(ind, other_parent)  # ind is modified in place
                    del ind.fitness.values
                    ind.gen_info.values['parent id'] = parents_id
                    do_mutation = False

            if do_mutation:
                # mutation
                self._toolbox.mutate(ind)

            del ind.fitness.values
            ind.gen_info.values['parent id'] = ind.gen_info.values['id']

            # update new individual's genetic info
            ind.gen_info.values['id'] = self._get_new_unique_ind_id()
            ind.gen_info.values['age'] = 0

        # bound genotype to given constraints
        if bound_genotype_thresh is not None:
            self._bound_inds_genotype(bound_genotype_thresh)

    def _replace_attr_based(self, src_pop, attr):
        """ Selects pop_size individuals from src_pop based on attr criteria, and store references to those individuals
        in the current object individuals.
        """
        assert attr in ['fitness', 'novelty']
        #pdb.set_trace()
        replacing_inds = self._toolbox.replace(src_pop, self._max_pop_size, fit_attr=attr)
        #replacing_inds = self._toolbox.replace(src_pop.inds, self._max_pop_size, fit_attr=attr)
        self.set_all_individuals(replacing_inds)

    def replace_n_multi_bd_tournsize_based(self, src_pop_inds, bd_filters, tournsize_ratio, putback=False):
        """Selection pop_size individuals from src_pop based on n_multi_bd_tournsize and stores references to the
        selected individuals into current object individuals."""

        nb_inds2generate = self._max_pop_size

        replacing_inds = select_n_multi_bd_tournsize(pop=src_pop_inds, nb_inds2generate=nb_inds2generate,
                                                     tournsize_ratio=tournsize_ratio, bd_filters=bd_filters,
                                                     putback=putback)

        assert len(replacing_inds) == self._max_pop_size

        self.set_all_individuals(replacing_inds)

    def replace_random_sample(self, src_pop):
        """Randomly selects pop_size individuals from src_pop and stores references to the selected individuals into
        current object individuals."""

        replacing_inds = random.sample(src_pop, self._max_pop_size)
        #replacing_inds = src_pop.random_sample(n_sample=self._max_pop_size, inplace=False)
        self.set_all_individuals(replacing_inds)

    def replace_novelty_based(self, src_pop):
        self._replace_attr_based(src_pop=src_pop, attr='novelty')

    def replace_fitness_based(self, src_pop):
        self._replace_attr_based(src_pop=src_pop, attr='fitness')


    def increment_age(self):
        """Increment the age of each element in the population."""
        for ind in self._inds:
            ind.gen_info.values['age'] += 1

    def get_successful_inds(self):
        return get_successul_inds_from_set(ind_set=self._inds)

    def get_n_successful_inds(self):
        return len(self.get_successful_inds())


