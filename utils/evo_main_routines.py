
import pdb
from utils.constants import SelectOffspringStrategy
from algorithms.population import Population
import numpy as np

from algorithms.archives.novelty_archive import NoveltyArchive
from algorithms.archives.elite_structured_archive import EliteStructuredArchive
from algorithms.archives.elite_novelty_structured_archive import EliteNoveltyStructuredArchive
from algorithms.archives.dummy_archive import DummyArchive

import utils.constants as consts


def select_quality_multi_bd_local_competition(novelty_metric, toolbox, ref_pop_inds, pop_size):
    n_bds = len(novelty_metric)
    bds_ids = [*range(n_bds)]
    safeguard_i_no_valid, max_trials_without_valid_bd = 0, 500
    n_selected_inds = 0

    off_inds = []
    while n_selected_inds < pop_size:
        i_sampled_bd = np.random.choice(bds_ids)
        candidates_inds = []
        for ind in ref_pop_inds:
            not_eligible_bd = ind.novelty.values[i_sampled_bd] is None
            if not_eligible_bd:
                continue
            ind.fitness.values = (ind.novelty.values[i_sampled_bd], ind.info.values['local_quality'][i_sampled_bd])
            candidates_inds.append(ind)

        no_valid_inds = len(candidates_inds) == 0
        if no_valid_inds:
            safeguard_i_no_valid += 1
            if safeguard_i_no_valid > max_trials_without_valid_bd:
                raise RuntimeError(f'No valid bd after {max_trials_without_valid_bd} trials.'
                                   f'(BDs might never be eligible).')
            continue

        safeguard_i_no_valid = 0

        off_ind = toolbox.select(ref_pop_inds, 1)[0]  # applies NSGA-II
        off_inds.append(off_ind)
        n_selected_inds += 1

    assert len(off_inds) == pop_size

    return off_inds



def select_novelty_search_local_competition(ref_pop_inds, pop_size, toolbox):
    for ind in ref_pop_inds:
        ind.fitness.values = (ind.novelty.values[0], ind.info.values['local_quality'])

    off_inds = toolbox.select(ref_pop_inds, k=pop_size)  # applies NSGA-II
    return off_inds


def select_off_inds(pop, nb_offsprings_to_generate, bd_filters, evo_process):
    """Offspring selection function."""

    if consts.RANDOM_SEL_FLG:
        off_inds = pop.random_sample(nb_offsprings_to_generate)

    else:
        # Selection = tournament
        if evo_process == 'classic_ea':
            # classical EA: selection on fitness
            off_inds = pop.select_tournament_attr(n_select=nb_offsprings_to_generate, attr='fitness')
        elif evo_process == 'ns_rand_multi_bd':
            off_inds = pop.select_n_multi_bd_tournsize(n_select=nb_offsprings_to_generate,
                                                       tournsize=consts.TOURNSIZE,
                                                       bd_filters=bd_filters)
        elif evo_process == 'random_search':
            off_inds = pop.random_sample(nb_offsprings_to_generate)
        else:
            # novelty search: selection on novelty
            pdb.set_trace()
            off_inds = pop.select_tournament_attr(n_select=nb_offsprings_to_generate, attr='novelty')

    return off_inds


def select_offspring_routine(pop, ref_pop_inds, novelty_metric, nb_offsprings_to_generate, bd_filters, evo_process,
                             toolbox, algo_variant, archive, pop_size, id_counter, prob_cx, **kwargs):

    oss = get_offspring_selection_strategy(algo_variant)

    if oss == SelectOffspringStrategy.RANDOM_FROM_POP:
        off_inds = select_off_inds(
            pop=pop,
            nb_offsprings_to_generate=nb_offsprings_to_generate,
            bd_filters=bd_filters,
            evo_process=evo_process
        )

    elif oss == SelectOffspringStrategy.RANDOM_FROM_ARCHIVE:
        off_inds = archive.random_sample(n_sample=pop_size)

    elif oss == SelectOffspringStrategy.NOVELTY_FROM_ARCHIVE:
        off_inds = archive.select_novelty_based(n_sample=pop_size)

    elif oss == SelectOffspringStrategy.FITNESS_FROM_ARCHIVE:
        off_inds = archive.select_fitness_based(n_sample=pop_size)

    elif oss == SelectOffspringStrategy.NOVELTY_SUCCESS_FROM_ARCHIVE:
        off_inds = archive.select_novelty_success_based(n_sample=pop_size)

    elif oss == SelectOffspringStrategy.NOVELTY_FITNESS_FROM_ARCHIVE:
        off_inds = archive.select_novelty_fitness_based(n_sample=pop_size, toolbox=toolbox)

    elif oss == SelectOffspringStrategy.NSLC:
        off_inds = select_novelty_search_local_competition(
            ref_pop_inds=ref_pop_inds, pop_size=pop_size, toolbox=toolbox
        )
    elif oss == SelectOffspringStrategy.SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE:
        off_inds = archive.select_success_based(n_sample=pop_size)

    elif oss == SelectOffspringStrategy.FORCE_SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE:
        off_inds = archive.select_force_success_based(n_sample=pop_size)

    else:
        raise NotImplementedError()

    off = Population(inds=off_inds, toolbox=toolbox, id_counter=id_counter, prob_cx=prob_cx, cx_flg=True)
    off.clone_individuals(inplace=True)

    return off


def update_pop_id_counter(pop, off):
    """Update id_counter with the current offspring id_counter.

    Mutating offspring individuals imply the creation of new individuals, modifying the global id_counter which is
    used to associate a unique id to each individual. Call this func after having mutated offspring to make sure
    pop.id_counter is always the reference attribute for setting unique ids."""
    updated_id_counter = off.id_counter
    pop.id_counter = updated_id_counter


def mutate_offspring(off, bound_genotype_thresh, evo_process, outcome_archive, mut_flg, gen, robot_name,
                     obj_vertices_poses, stabilized_obj_pose):

    for ind in off.inds:
        ind.gen_info.values['parent_genome'] = np.array(ind)

    if mut_flg == 'gauss':
        off.var_and(bound_genotype_thresh=bound_genotype_thresh, evo_process=evo_process)

    elif mut_flg == 'serene':
        # standard gaussian mut
        off.var_and(bound_genotype_thresh=bound_genotype_thresh, evo_process=evo_process)

    else:
        raise AttributeError(f'Unknown mutation flag: mut_flg={mut_flg}')


def mutate_offspring_routine(
        off,
        bound_genotype_thresh,
        evo_process,
        outcome_archive,
        mut_flg,
        pop,
        gen,
        robot_name,
        obj_vertices_poses,
        stabilized_obj_pose,
        **kwargs
):
    mutate_offspring(
        off=off,
        bound_genotype_thresh=bound_genotype_thresh,
        evo_process=evo_process,
        outcome_archive=outcome_archive,
        mut_flg=mut_flg,
        gen=gen,
        robot_name=robot_name,
        obj_vertices_poses=obj_vertices_poses,
        stabilized_obj_pose=stabilized_obj_pose,
    )

    off.update_scs2scs_metrics()

    update_pop_id_counter(pop=pop, off=off)


def get_offspring_selection_strategy(algo_variant):

    if algo_variant in consts.POP_BASED_RANDOM_SELECTION_ALGO_VARIANTS:
        oss = SelectOffspringStrategy.RANDOM_FROM_POP

    elif algo_variant in consts.ARCHIVE_BASED_RANDOM_SELECTION_ALGO_VARIANTS:
        oss = SelectOffspringStrategy.RANDOM_FROM_ARCHIVE

    elif algo_variant in consts.ARCHIVE_BASED_NOVELTY_SELECTION_ALGO_VARIANTS:
        oss = SelectOffspringStrategy.NOVELTY_FROM_ARCHIVE

    elif algo_variant in consts.ARCHIVE_BASED_FITNESS_SELECTION_ALGO_VARIANTS:
        oss = SelectOffspringStrategy.FITNESS_FROM_ARCHIVE

    elif algo_variant in consts.ARCHIVE_BASED_NOVELTY_SUCCESS_SELECTION_ALGO_VARIANTS:
        oss = SelectOffspringStrategy.NOVELTY_SUCCESS_FROM_ARCHIVE

    elif algo_variant in consts.ARCHIVE_BASED_NOVELTY_FITNESS_SELECTION_ALGO_VARIANTS:
        oss = SelectOffspringStrategy.NOVELTY_FITNESS_FROM_ARCHIVE

    elif algo_variant in consts.NSLC_NSGA_II_SELECTION_ALGO_VARIANTS:
        oss = SelectOffspringStrategy.NSLC

    elif algo_variant in consts.ELITE_STRUCTURED_ARCHIVE_SUCCESS_BASED_SELECTION_ALGO_VARIANTS:
        oss = SelectOffspringStrategy.SUCCESS_BASED_FROM_STRUCTURED_ARCHIVE

    elif algo_variant in consts.SERENE_QD_VARIANTS:
        oss = SelectOffspringStrategy.RANDOM_FROM_POP

    else:
        raise NotImplementedError()

    return oss


def init_archive(algo_variant, archive_kwargs):

    assert algo_variant in consts.SUPPORTED_VARIANTS_NAMES

    if algo_variant in consts.NOVELTY_ARCHIVE_ALGO_VARIANTS:
        archive_class = NoveltyArchive

    elif algo_variant in consts.ELITE_STRUCTURED_ARCHIVE_ALGO_VARIANTS:
        archive_class = EliteStructuredArchive

    elif algo_variant in consts.ELITE_NOVELTY_STRUCTURED_ARCHIVE_ALGO_VARIANTS:
        archive_class = EliteNoveltyStructuredArchive

    elif algo_variant in consts.ARCHIVE_LESS_ALGO_VARIANTS:
        archive_class = DummyArchive

    elif algo_variant in consts.PYRIBS_QD_VARIANTS:
        archive_class = None  # until refactoring

    elif algo_variant in consts.SERENE_QD_VARIANTS:
        archive_class = NoveltyArchive

    else:
        raise NotImplementedError()

    return archive_class(**archive_kwargs)




