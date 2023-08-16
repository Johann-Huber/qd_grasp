import pdb

import sys
from tqdm import tqdm, trange

from ribs.schedulers import Scheduler
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter

from algorithms.archives.outcome_archive import OutcomeArchive

from utils.progression_monitoring import ProgressionMonitoring

from algorithms.population import Population

from utils.io_run_data import dump_archive_success_routine, export_running_data_routine
import utils.constants as consts
import numpy as np

from algorithms.archives import structured_archive as sa


def get_cma_mae_pyribs_learning_rate(algo_variant):
    if algo_variant in consts.CMA_MAE_ALGO_VARIANTS:
        return consts.CMA_MAE_PREDEFINED_ALPHA
    elif algo_variant in consts.CMA_ME_ALGO_VARIANTS:
        return consts.CMA_ME_PREDEFINED_ALPHA
    elif algo_variant in consts.CMA_ES_ALGO_VARIANTS:
        return consts.CMA_ES_PREDEFINED_ALPHA
    else:
        raise NotImplementedError()


def init_archive_pyribs(solution_dim, algo_variant):
    lr = get_cma_mae_pyribs_learning_rate(algo_variant)

    archive_kwargs = {
        'solution_dim': solution_dim,
        'dims': (sa.N_BINS_PER_DIM_POS_X_TOUCH, sa.N_BINS_PER_DIM_POS_Y_TOUCH, sa.N_BINS_PER_DIM_POS_Z_TOUCH),
        'ranges': [
            (sa.MIN_X_TOUCH_VAL, sa.MAX_X_TOUCH_VAL),
            (sa.MIN_Y_TOUCH_VAL, sa.MAX_Y_TOUCH_VAL),
            (sa.MIN_Z_TOUCH_VAL, sa.MAX_Z_TOUCH_VAL)
        ],
        'learning_rate': lr,
        'threshold_min': -1.0 #0.0 # for null fit to be added
    }
    return GridArchive(**archive_kwargs)


def evaluate_grasp_pyribs(toolbox, evaluate_fn, inds):

    evaluation_pop = list(toolbox.map(evaluate_fn, inds))

    b_descriptors, is_scs_fitnesses, infos = map(list, zip(*evaluation_pop))

    #objective_batch = np.array(fitnesses)[:, 0].tolist()  # required shape for tell(): (batch_size,)
    measure_batch = b_descriptors
    infos_batch = infos

    objective_batch = [
        ind_info['normalized_multi_fit'] if ind_info['is_success'] else 0. for ind_info in infos
    ]
    #print('objective_batch=', objective_batch)
    return objective_batch, measure_batch, infos_batch


def run_qd_pyribs(
    toolbox,
    evaluate_fn,
    stats_tracker,
    genotype_len,
    n_budget_rollouts,
    reinit_research_flg,
    outcome_archive_kwargs,
    bound_genotype_thresh,
    prob_cx,
    timer,
    run_name,
    run_details,
    algo_variant,
):
    progression_monitoring = ProgressionMonitoring(
        n_budget_rollouts=n_budget_rollouts, reinit_research_flg=reinit_research_flg
    )

    outcome_archive = OutcomeArchive(**outcome_archive_kwargs)

    archive = init_archive_pyribs(
        solution_dim=genotype_len,
        algo_variant=algo_variant,
    )
    random_init_solution = np.random.uniform(low=-1, high=1, size=genotype_len)
    emitters = [
        EvolutionStrategyEmitter(
            archive,
            x0=random_init_solution,  # initial solution (must be a single ind)
            sigma0=0.5,
            ranker="imp",
            selection_rule="mu",
            restart_rule="basic",
            batch_size=consts.CMA_MAE_EMITTER_BATCH_SIZE,
        ) for _ in range(consts.CMA_MAE_N_EMITTERS)
    ]
    scheduler = Scheduler(archive, emitters)

    gen, do_iterate_gen = 0, True
    while do_iterate_gen:
        gen += 1

        solution_batch = scheduler.ask()

        pop = Population(
            toolbox=toolbox,
            max_pop_size=consts.CMA_MAE_POP_SIZE,
            genotype_len=genotype_len,
            n_reinit_flg=reinit_research_flg,
            bound_genotype_thresh=bound_genotype_thresh,
            curr_n_evals=progression_monitoring.n_eval,
            prob_cx=prob_cx,
        )
        pop.set_genomes_to_deap_pop(genomes=solution_batch)

        objective_batch, measure_batch, infos_batch = evaluate_grasp_pyribs(
            toolbox=toolbox, evaluate_fn=evaluate_fn, inds=solution_batch
        )
        #pdb.set_trace()
        pop.update_individuals(
            fitnesses=[(fit,) for fit in objective_batch],
            b_descriptors=measure_batch,
            infos=infos_batch,
            curr_n_evals=progression_monitoring.n_eval
        )

        added_inds, scs_inds_generated = outcome_archive.update(pop)
        progression_monitoring.update(pop=pop, outcome_archive=outcome_archive)
        stats_tracker.update(pop=pop, outcome_archive=outcome_archive, curr_n_evals=progression_monitoring.n_eval, gen=gen)

        do_dump_scs_archive = consts.DUMP_SCS_ARCHIVE_ON_THE_FLY and gen % consts.N_GEN_FREQ_DUMP_SCS_ARCHIVE == 0
        if do_dump_scs_archive:
            dump_archive_success_routine(
                timer=timer,
                timer_label=consts.NS_RUN_TIME_LABEL,
                run_name=run_name,
                curr_neval=progression_monitoring.n_eval,
                outcome_archive=outcome_archive,
            )

        scheduler.tell(objective_batch, measure_batch)

        curr_n_eval = progression_monitoring.n_eval
        if curr_n_eval > n_budget_rollouts:
            print(f'curr_n_eval={curr_n_eval} > n_budget_rollouts={n_budget_rollouts} | End of running.')
            stats_tracker.n_generations = gen
            do_iterate_gen = False
            continue  # end of the evolutionary process

        pass  # end of generation
    #pdb.set_trace()

    print('\n\nEnd of running.')

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

