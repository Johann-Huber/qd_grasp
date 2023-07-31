
import numpy as np


class Evaluator:
    def __init__(self, evo_process, eval_func, toolbox):

        self._toolbox = toolbox  # deap toolbox for evolutionary algorithms utilities
        self._ind_evaluation_func = eval_func  # evaluation function applied on each ind in // to get their BDs & infos

        if evo_process == 'ns_rand_change_bd':
            raise NotImplementedError()

    @property
    def ind_evaluation_func(self):
        return self._ind_evaluation_func

    def evaluate(self, pop=None, inds=None):
        """Evaluate the given list of individual.

        :param inds: list of individuals to evaluate
        :return: b_descriptors : list containing behavior descriptor associated to each individual
        :return: fitnesses : list containing fitness associated to each individual
        :return: infos : list containing dicts for information associated to each individual
                         (keys : 'energy', 'reward', 'is_success', 'end effector position relative object',
                         'end effector xyzw relative object', 'diversity_descriptor')
        """

        assert pop is not None or inds is not None

        if pop is not None:
            raise NotImplementedError()

        evaluation_pop = list(self._toolbox.map(self._ind_evaluation_func, inds))
        # expacted pattern : [ (behavior, (fitness,), info)_1, ..., (behavior, (fitness,), info)_n_pop ]

        b_descriptors, fitnesses, infos = map(list, zip(*evaluation_pop))
        return b_descriptors, fitnesses, infos



