
import pdb

from external_pkg.serene.core.searcher import Searcher
from algorithms.archives.outcome_archive import OutcomeArchive

from utils.progression_monitoring import ProgressionMonitoring
from utils.io_run_data import dump_archive_success_routine, export_running_data_routine
from utils.evo_main_routines import init_archive

import utils.constants as consts


def init_searcher_params(qds_args):
    mutation_parameters = {
        'sigma': consts.SIGMA,
        'mut_prob': 0.3,
        'mu': 0,
    }

    return {
        'mutation_parameters': mutation_parameters,
        'evaluation_budget': qds_args['n_budget_rollouts'],
        'genome_size': qds_args['genotype_len'],
        'genome_limit': [-1, 1],
        'progression_monitoring': qds_args['progression_monitoring'],
        'n_budget_rollouts': qds_args['n_budget_rollouts'],
        'genotype_len': qds_args['genotype_len'],
        'toolbox': qds_args['toolbox'],
        'reinit_research_flg': qds_args['reinit_research_flg'],
        'bound_genotype_thresh': qds_args['bound_genotype_thresh'],
        'prob_cx': qds_args['prob_cx'],
        'pop_size': qds_args['pop_size']
    }


def run_qd_serene(
    n_budget_rollouts,
    genotype_len,
    toolbox,
    reinit_research_flg,
    bound_genotype_thresh,
    prob_cx,
    pop_size,
    evaluate_fn,
    outcome_archive_kwargs,
    stats_tracker,
    algo_variant,
    is_novelty_required,
    evo_process,
    bd_indexes,
    bd_filters,
    novelty_metric,
    bd_bounds,
    nb_offsprings_to_generate,
    mut_flg,
    robot_name,
    obj_vertices_poses,
    stabilized_obj_pose,
    archive_kwargs,
    run_name,
    timer,
    run_details,
):

    progression_monitoring = ProgressionMonitoring(
        n_budget_rollouts=n_budget_rollouts, reinit_research_flg=reinit_research_flg
    )

    archive = init_archive(algo_variant, archive_kwargs)

    outcome_archive = OutcomeArchive(**outcome_archive_kwargs)

    qds_args = {
        'progression_monitoring': progression_monitoring,
        'n_budget_rollouts': n_budget_rollouts,
        'genotype_len': genotype_len,
        'toolbox': toolbox,
        'reinit_research_flg': reinit_research_flg,
        'bound_genotype_thresh': bound_genotype_thresh,
        'prob_cx': prob_cx,
        'pop_size': pop_size,
        'evaluate_fn': evaluate_fn,
        'outcome_archive': outcome_archive,
        'archive': archive,
        'stats_tracker': stats_tracker,
        'algo_variant': algo_variant,
        'is_novelty_required': is_novelty_required,
        'evo_process': evo_process,
        'bd_indexes': bd_indexes,
        'bd_filters': bd_filters,
        'novelty_metric': novelty_metric,
        'bd_bounds': bd_bounds,
        'nb_offsprings_to_generate': nb_offsprings_to_generate,
        'mut_flg': mut_flg,
        'robot_name': robot_name,
        'obj_vertices_poses': obj_vertices_poses,
        'stabilized_obj_pose': stabilized_obj_pose,
        'outcome_archive_kwargs': outcome_archive_kwargs,
        'run_name': run_name,
        'timer': timer,

    }
    id_counter = 0

    serene_params = init_searcher_params(qds_args)
    searcher = Searcher(serene_params, id_counter=id_counter)
    id_counter = searcher.population.id_counter

    evaluated_points = 0
    while evaluated_points < n_budget_rollouts:

        print("Generation: {}".format(searcher.generation))

        id_counter = searcher.chunk_step(qds_args=qds_args, searcher_params=serene_params, id_counter=id_counter)

        evaluated_points = qds_args['progression_monitoring'].n_eval

    print('end of run serene')
    pop = searcher.population
    gen = searcher.generation

    print('\nLast dump of success archive...')
    dump_archive_success_routine(
        timer=timer,
        timer_label=consts.NS_RUN_TIME_LABEL,
        run_name=run_name,
        curr_neval=progression_monitoring.n_eval,
        outcome_archive=outcome_archive,
        is_last_call=True,
    )
    print(f'\nLatest success archive has been successfully dumped to {run_name}')

    timer.stop(label=consts.NS_RUN_TIME_LABEL)

    run_infos = {
        f'Number of triggered outcome cells (outcome_archive len, out of {outcome_archive.get_max_size()})':
            len(outcome_archive),
        'Number of successful individuals (outcome_archive get_n_successful_cells)':
            outcome_archive.get_n_successful_cells(),
        'Number of evaluations (progression_monitoring.n_eval)': progression_monitoring.n_eval,
        'Number of attempting population reinitialization': pop.n_reinitialization,
        'elapsed_time': timer.get_all(format='h:m:s'),
        'Number of computed generations (gen)': gen,
        'first_saved_ind_gen': stats_tracker.first_saved_ind_gen,
        'first_saved_ind_n_evals': stats_tracker.first_saved_ind_n_evals,
    }

    export_running_data_routine(
        stats_tracker=stats_tracker,
        run_name=run_name,
        run_details=run_details,
        run_infos=run_infos
    )

    if pop.is_empty():
        print('Warning empty pop.')

    success_archive = outcome_archive.get_successful_inds()
    return success_archive






