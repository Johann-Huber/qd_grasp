
from deap import tools, base, creator
from scoop import futures
import random

import utils.constants as consts


def init_deap_creator(algo_variant, evo_process):
    """Initialize DEAP creator, which defines utility classes.
    DEAP is a third party library that provide usefull tools to implement evolutionary algorithms.
    See: https://deap.readthedocs.io/"""

    creator.create('BehaviorDescriptor', list)
    creator.create('GenInfo', dict)
    creator.create('Info', dict)

    if evo_process in consts.MULTI_BD_EVO_PROCESSES:
        creator.create('Novelty', list)
    else:
        creator.create('Novelty', base.Fitness, weights=(1.0,))

    if evo_process in consts.EVO_PROCESS_WITH_LOCAL_COMPETITION:
        creator.create('Fit', base.Fitness, weights=(1.0, 1.0))  # (novelty, local_quality)
    elif algo_variant in consts.ARCHIVE_BASED_NOVELTY_FITNESS_SELECTION_ALGO_VARIANTS:
        creator.create('Fit', base.Fitness, weights=(1.0, 1.0))  # (novelty, normalized_fitness)
    else:
        creator.create('Fit', base.Fitness, weights=(-1.0,))

    creator.create('Individual', list, behavior_descriptor=creator.BehaviorDescriptor,
                   novelty=creator.Novelty, fitness=creator.Fit, info=creator.Info,
                   gen_info=creator.GenInfo)


def init_deap_toolbox(algo_variant, genotype_len):
    """Initialize DEAP toolbox, which defines utility functions.
    DEAP is a third party library that provide usefull tools to implement evolutionary algorithms.
    See: https://deap.readthedocs.io/"""

    toolbox = base.Toolbox()

    # Scoop parallelization
    toolbox.register('map', futures.map)

    # Initialization functions
    toolbox.register('init_ind', random.uniform, -1, 1)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.init_ind, genotype_len)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Variation operators
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=consts.SIGMA, indpb=0.3)
    toolbox.register('select', tools.selTournament, tournsize=consts.TOURNSIZE)

    is_there_nsga2_sel = algo_variant in consts.ARCHIVE_BASED_NOVELTY_FITNESS_SELECTION_ALGO_VARIANTS or \
                         algo_variant in consts.NSLC_ALGO_VARIANTS
    if is_there_nsga2_sel:
        toolbox.register('select', tools.selNSGA2)
    else:
        toolbox.register('select', tools.selTournament, tournsize=consts.TOURNSIZE)

    toolbox.register('replace', tools.selBest)  # confusing name : replace is a selection func
    return toolbox


def initialize_deap_tools(genotype_len, evo_process, algo_variant):
    """Initialize DEAP tools.
    DEAP is a third party library that provide usefull tools to implement evolutionary algorithms.
    See: https://deap.readthedocs.io/"""
    init_deap_creator(algo_variant=algo_variant, evo_process=evo_process)
    toolbox = init_deap_toolbox(algo_variant=algo_variant, genotype_len=genotype_len)
    return toolbox
