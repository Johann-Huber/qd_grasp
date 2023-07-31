
import random
from deap import tools, base
import pdb
import utils.constants as consts
from algorithms.population import Population
from algorithms.evaluator import Evaluator
from utils.progression_monitoring import ProgressionMonitoring

from utils.evo_tools import replace_pop, compute_bd_filters
from utils.evo_main_routines import mutate_offspring_routine, \
    select_offspring_routine, select_off_inds, init_archive
from utils.novelty_computation import assess_novelties_and_local_quality_single_bd_vec, assess_novelties
from utils.io_run_data import dump_archive_success_routine, export_running_data_routine

from algorithms.stats.stats_tracker import StatsTracker
from utils.timer import Timer
from scoop import futures

from algorithms.archives.outcome_archive import OutcomeArchive

from algorithms.archives.utils import get_fill_archive_strat

from algorithms.pyribs_qd_interface import run_qd_pyribs
from algorithms.serene_qd_interface import run_qd_serene


creator = None
def set_creator(cr):
    global creator
    creator = cr


def initialize_deap_tools(genotype_len, pop_size, parallelize, evo_process, algo_variant, n_genes_per_keypoint):
    """Initialize the toolbox"""

    global creator
    toolbox = base.Toolbox()

    assert parallelize
    # scoop parallelization
    toolbox.register('map', futures.map)

    # initialization functions
    toolbox.register('init_ind', random.uniform, -1, 1)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.init_ind, genotype_len)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # operators
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
    return creator, toolbox


def select_and_clone_offspring(pop, nb_offsprings_to_generate, bd_filters, evo_process, toolbox, id_counter):
    """Select offsrping from pop and return the Population instance containing the cloned offspring.
    The output population off is read to be mutated."""

    off_inds = select_off_inds(pop=pop, nb_offsprings_to_generate=nb_offsprings_to_generate,
                               bd_filters=bd_filters, evo_process=evo_process)
    off = Population(inds=off_inds, toolbox=toolbox, id_counter=id_counter)
    off.clone_individuals(inplace=True)
    return off


def update_novelty_routine(is_novelty_required, inds2update, archive, evo_process, bd_indexes, bd_filters,
                           novelty_metric, algo_variant, **kwargs):

    if is_novelty_required:

        if algo_variant in consts.SINGLE_BD_NSLC_ALGO_VARIANTS:
            #  both novelty and local quality must be computed at the same time to avoid multiple KNN calls
            novelties_inds2update, local_qualities_inds2update = \
                assess_novelties_and_local_quality_single_bd_vec(
                    pop=inds2update,
                    archive=archive.inds,
                    novelty_metric=novelty_metric
                )
            Population.update_novelties_inds(inds=inds2update, novelties=novelties_inds2update)
            Population.update_local_qualities_inds(inds=inds2update, local_qualities=local_qualities_inds2update)

        else:
            novelties_inds2update = assess_novelties(
                pop=inds2update,
                archive=archive.inds,
                evo_process=evo_process,
                bd_indexes=bd_indexes,
                bd_filters=bd_filters,
                novelty_metric=novelty_metric
            )
            Population.update_novelties_inds(inds=inds2update, novelties=novelties_inds2update)

    else:
        novelties_inds2update = None

    return novelties_inds2update

def get_missing_ns_raw_args(ns_raw_args):
    assert isinstance(ns_raw_args, dict)

    expected_raw_args = {'evaluation_function', 'genotype_len', 'bd_bounds', 'algo_variant', 'evo_process',
                         'bound_genotype_thresh', 'pop_size', 'measures', 'bd_indexes', 'archive_limit_size', 'nb_cells',
                         'novelty_metric', 'n_saved_ind_early_stop', 'run_name',
                         'n_budget_rollouts', 'mut_flg', 'bd_flg', 'parallelize', 'archive_limit_strat'}

    missing_keys = {key for key in expected_raw_args if key not in ns_raw_args}
    return missing_keys


def sanity_check_qd_raw_args(ns_raw_args, verbose=True):

    missing_keys = get_missing_ns_raw_args(ns_raw_args)
    if len(missing_keys) > 0:
        raise AttributeError(f'missing key in raw ns args: {missing_keys}')


    algo_variant = ns_raw_args['algo_variant']
    if algo_variant not in consts.SUPPORTED_VARIANTS_NAMES:
        raise AttributeError(f'not supported algo_variant: {algo_variant} '
                             f'(supported : {consts.SUPPORTED_VARIANTS_NAMES})')


    run_name = ns_raw_args['run_name']
    if type(run_name) not in consts.SUPPORTED_DUMP_PATH_TYPES:
        raise AttributeError(f'not supported run_name type: type={type(run_name)} '
                             f'(supported : {consts.SUPPORTED_DUMP_PATH_TYPES})')

    mut_flg = ns_raw_args['mut_flg']
    if mut_flg not in consts.SUPPORTED_MUT_FLGS:
        raise AttributeError(f'not supported mut_flg: {mut_flg} '
                             f'(supported : {consts.SUPPORTED_MUT_FLGS})')

    if verbose:
        print('sanity_check_qd_raw_args ok')


def init_run_details(ns_raw_args):
    evaluation_function = ns_raw_args['evaluation_function']
    genotype_len = ns_raw_args['genotype_len']
    bd_bounds = ns_raw_args['bd_bounds']
    algo_variant = ns_raw_args['algo_variant']
    evo_process = ns_raw_args['evo_process']
    bound_genotype_thresh = ns_raw_args['bound_genotype_thresh']
    pop_size = ns_raw_args['pop_size']
    bd_indexes = ns_raw_args['bd_indexes']
    archive_limit_size = ns_raw_args['archive_limit_size']
    archive_limit_strat = ns_raw_args['archive_limit_strat']
    nb_cells = ns_raw_args['nb_cells']
    mut_flg = ns_raw_args['mut_flg']
    bd_flg = ns_raw_args['bd_flg']
    reinit_research_flg = ns_raw_args['reinit_research_flg']
    nb_offsprings_to_generate = ns_raw_args['nb_offsprings_to_generate']
    bd_filters_str = str(ns_raw_args['bd_filters'])
    is_novelty_required = ns_raw_args['is_novelty_required']
    novelty_metric = ns_raw_args['novelty_metric']

    env_name = ns_raw_args['scene_details']['env_name']
    targeted_object = ns_raw_args['scene_details']['targeted_object']
    controller_type = ns_raw_args['scene_details']['controller_type']
    close_while_touch = ns_raw_args['scene_details']['close_while_touch']

    run_details = {
        'Evaluation function': evaluation_function.__name__,
        'Genotype length (genotype_len)': genotype_len,
        'BD boundaries (bd_bounds)': bd_bounds,
        'Algorithm variant (algo_variant)': algo_variant,
        'Evolutionary process (evo_process)': evo_process,
        'Genotype boundaries (bound_genotype_thresh)': bound_genotype_thresh,
        'Population size (pop_size)': pop_size,
        'BD indices (bd_indexes)': bd_indexes,
        'Max number of individuals in the archive (archive_limit_size)': archive_limit_size,
        'Archive removal strategy when max size is reached (archive_limit_strat)': archive_limit_strat,
        'Number of cells (nb_cells)': nb_cells,
        'Mutation flag (mut_flg)': mut_flg,
        'Behavioral descriptors flag (bd_flg)': bd_flg,
        'Reinitialize population flag (reinit_research_flg)': reinit_research_flg,
        'Number of generated offspring per generation (nb_offsprings_to_generate)': nb_offsprings_to_generate,
        'Filters to compute BDs from the concatenated BD vector (bd_filters)': bd_filters_str,
        'Novelty required flag (is_novelty_required)': is_novelty_required,
        'Novelty metric (novelty_metric)': novelty_metric,
        'Environment name' : env_name,
        'Targeted object': targeted_object,
        'Controller type': controller_type,
        'Close while touch': close_while_touch,
    }

    return run_details


def init_archive_kwargs(ns_raw_args):
    if ns_raw_args['algo_variant'] in consts.NOVELTY_ARCHIVE_ALGO_VARIANTS:
        archive_kwargs = {
            'archive_limit_size': ns_raw_args['archive_limit_size'],
            'archive_limit_strat': ns_raw_args['archive_limit_strat'],
            'pop_size': ns_raw_args['pop_size'],
        }
    elif ns_raw_args['algo_variant'] in consts.ELITE_STRUCTURED_ARCHIVE_ALGO_VARIANTS:
        archive_kwargs = {}
    elif ns_raw_args['algo_variant'] in consts.ELITE_NOVELTY_STRUCTURED_ARCHIVE_ALGO_VARIANTS:
        archive_kwargs = {
            'novelty_metric': ns_raw_args['novelty_metric'],
        }
    elif ns_raw_args['algo_variant'] in consts.ARCHIVE_LESS_ALGO_VARIANTS:
        archive_kwargs = {}
    elif ns_raw_args['algo_variant'] in consts.PYRIBS_QD_VARIANTS:
        archive_kwargs = {}  # until refactoring
    elif ns_raw_args['algo_variant'] in consts.SERENE_QD_VARIANTS:
        archive_kwargs = {
            'archive_limit_size': ns_raw_args['archive_limit_size'],
            'archive_limit_strat': ns_raw_args['archive_limit_strat'],
            'pop_size': ns_raw_args['pop_size'],
        }
    else:
        raise NotImplementedError()

    is_pyribs_algo = ns_raw_args['algo_variant'] in consts.PYRIBS_QD_VARIANTS
    if not is_pyribs_algo:
        archive_kwargs['fill_archive_strat'] = get_fill_archive_strat(ns_raw_args['evo_process'])
        archive_kwargs['bd_flg'] = ns_raw_args['bd_flg']

    return archive_kwargs


def init_outcome_archive_kwargs(ns_raw_args):
    outcome_archive_kwargs = {
        'bd_flg': ns_raw_args['bd_flg'],
    }
    return outcome_archive_kwargs


def add_missing_qd_args(ns_raw_args):

    ns_raw_args['nb_offsprings_to_generate'] = int(ns_raw_args['pop_size'] * consts.OFFSPRING_NB_COEFF)
    ns_raw_args['bd_filters'] = compute_bd_filters(ns_raw_args['bd_indexes'])
    ns_raw_args['is_novelty_required'] = ns_raw_args['evo_process'] not in consts.NO_NOVELTY_EVO_PROCESSES

    ns_raw_args['reinit_research_flg'] = ns_raw_args['algo_variant'] in consts.VARIANTS_WITH_EVO_PROC_REINIT

    ns_raw_args['run_details'] = init_run_details(ns_raw_args)

    ns_raw_args['archive_kwargs'] = init_archive_kwargs(ns_raw_args)
    ns_raw_args['outcome_archive_kwargs'] = init_outcome_archive_kwargs(ns_raw_args)


def run_qd_wrapper(algo_args):

    sanity_check_qd_raw_args(ns_raw_args=algo_args)

    add_missing_qd_args(algo_args)

    success_archive = run_qd(**algo_args)

    return success_archive


def initialize_pop_and_archive(
    toolbox,
    pop_size,
    prob_cx,
    genotype_len,
    reinit_research_flg,
    bound_genotype_thresh,
    closer_genome_init_func,
    algo_variant,
    archive_kwargs,
    outcome_archive_kwargs,
    evaluator,
    is_novelty_required,
    evo_process,
    bd_indexes,
    bd_filters,
    novelty_metric,
    bd_bounds,
    progression_monitoring,
    stats_tracker,
    id_counter=None,
    gen=0,
    **kwargs,
):
    DEBUG_SINGLE_IND_FLG = False

    pop = Population(
        toolbox=toolbox,
        max_pop_size=pop_size,
        genotype_len=genotype_len,
        n_reinit_flg=reinit_research_flg,
        bound_genotype_thresh=bound_genotype_thresh,
        closer_genome_init_func=closer_genome_init_func,
        curr_n_evals=progression_monitoring.n_eval,
        prob_cx=prob_cx,
    )

    if DEBUG_SINGLE_IND_FLG:
        pdb.set_trace()
        genome = None
        pop.set_all_inds_from_same_genome_debug(genome)

    archive = init_archive(algo_variant, archive_kwargs)

    outcome_archive = OutcomeArchive(**outcome_archive_kwargs)

    pop.evaluate_and_update_inds(evaluator=evaluator)

    novelty_pop = update_novelty_routine(
        is_novelty_required=is_novelty_required,
        inds2update=pop.inds,
        archive=archive,
        evo_process=evo_process,
        bd_indexes=bd_indexes,
        bd_filters=bd_filters,
        novelty_metric=novelty_metric,
        algo_variant=algo_variant,
        bd_bounds=bd_bounds
    )

    if algo_variant not in consts.PYRIBS_QD_VARIANTS:
        archive.fill(
            pop2add=pop,
            novelties=novelty_pop
        )

    added_inds, scs_inds_generated = outcome_archive.update(pop)
    progression_monitoring.update(pop=pop, outcome_archive=outcome_archive)

    stats_tracker.update(pop=pop, outcome_archive=outcome_archive, curr_n_evals=progression_monitoring.n_eval, gen=gen)

    ref_pop_inds = pop.inds

    if id_counter is None:
        id_counter = archive.get_last_id_counter() + 1

    return pop, archive, outcome_archive, progression_monitoring, stats_tracker, ref_pop_inds, id_counter


def run_qd_local(
    toolbox,
    evaluator,
    stats_tracker,
    novelty_metric,
    evaluation_function,
    genotype_len,
    bd_bounds,
    prob_cx,
    algo_variant,
    evo_process,
    bound_genotype_thresh,
    pop_size,
    measures,
    bd_indexes,
    archive_limit_size,
    archive_limit_strat,
    nb_cells,
    n_saved_ind_early_stop,
    run_name,
    n_budget_rollouts,
    mut_flg,
    parallelize,
    run_details,  # preprocessed
    nb_offsprings_to_generate,  # preprocessed
    bd_filters,  # preprocessed
    is_novelty_required,  # preprocessed
    reinit_research_flg,
    closer_genome_init_func,
    controller_info,
    robot_name,
    timer,
    obj_vertices_poses,
    stabilized_obj_pose,
    archive_kwargs,  # keyword args associted to archive type
    outcome_archive_kwargs,  # keyword args associated to output archive
    **kwargs
):

    #for k in kwargs:
    #    print(f'Unused given key: {k}')

    research_must_be_reinit = False  # flag that triggers the evolutionary process reinitialization
    n_gen_rolling_reinit_research = 0

    progression_monitoring = ProgressionMonitoring(
        n_budget_rollouts=n_budget_rollouts, reinit_research_flg=reinit_research_flg
    )

    fixed_attr_dict = {
        'toolbox': toolbox,
        'pop_size': pop_size,
        'genotype_len': genotype_len,
        'reinit_research_flg': reinit_research_flg,
        'bound_genotype_thresh': bound_genotype_thresh,
        'closer_genome_init_func': closer_genome_init_func,
        'algo_variant': algo_variant,
        'archive_kwargs': archive_kwargs,
        'outcome_archive_kwargs': outcome_archive_kwargs,
        'evaluator': evaluator,
        'is_novelty_required': is_novelty_required,
        'evo_process': evo_process,
        'bd_indexes': bd_indexes,
        'bd_filters': bd_filters,
        'novelty_metric': novelty_metric,
        'bd_bounds': bd_bounds,
        'nb_offsprings_to_generate': nb_offsprings_to_generate,
        'mut_flg': mut_flg,
        'prob_cx': prob_cx,
        'robot_name': robot_name,
        'obj_vertices_poses': obj_vertices_poses,
        'stabilized_obj_pose': stabilized_obj_pose,
    }

    # --------------------------------------- INITIALIZATION -----------------------------------------

    pop, archive, outcome_archive, progression_monitoring, stats_tracker, ref_pop_inds, id_counter = \
        initialize_pop_and_archive(
            progression_monitoring=progression_monitoring,
            stats_tracker=stats_tracker,
            **fixed_attr_dict
    )

    # --------------------------------------- BEGIN EVOLUTION ----------------------------------------

    gen, do_iterate_gen = 0, True
    while do_iterate_gen:
        gen += 1

        is_early_stop_flg = 0 <= n_saved_ind_early_stop <= len(outcome_archive)
        if is_early_stop_flg:
            stats_tracker.set_details(key='nb of generations', val=gen)
            break

        n_gen_rolling_reinit_research += 1
        if n_gen_rolling_reinit_research > consts.N_GEN_NO_SUCCESS_REINIT_RESEARCH_FREQ and \
                len(outcome_archive.get_successful_inds()) == 0:
            research_must_be_reinit = True
            n_gen_rolling_reinit_research = 0

        if reinit_research_flg and research_must_be_reinit:
            print('REINITIALIZATION')
            pop, archive, outcome_archive, progression_monitoring, stats_tracker, ref_pop_inds, id_counter = \
                initialize_pop_and_archive(
                    progression_monitoring=progression_monitoring,
                    stats_tracker=stats_tracker,
                    id_counter=id_counter,
                    gen=gen,
                    **fixed_attr_dict
                )
            research_must_be_reinit = False

        # ------------------------------------------ SELECT ------------------------------------------

        off = select_offspring_routine(
            pop=pop,
            ref_pop_inds=ref_pop_inds,
            archive=archive,
            id_counter=id_counter,
            **fixed_attr_dict
        )

        # ------------------------------------------ MUTATE ------------------------------------------

        mutate_offspring_routine(
            off=off,
            outcome_archive=outcome_archive,
            pop=pop,
            gen=gen,
            **fixed_attr_dict
        )
        id_counter = off.id_counter

        # ------------------------------------------ EVALUATE ----------------------------------------

        if algo_variant in consts.POP_BASED_ALGO_VARIANTS:
            ref_pop_inds = pop.inds + off.inds
            invalid_inds = pop.get_invalid_inds() + off.get_invalid_inds()
        else:
            ref_pop_inds = off.inds
            invalid_inds = off.get_invalid_inds()

        inv_pop = Population(
            inds=invalid_inds, toolbox=toolbox, genotype_len=genotype_len, id_counter=pop.id_counter, prob_cx=prob_cx
        )
        inv_pop.evaluate_and_update_inds(evaluator=evaluator)

        novelty_updated_inds = update_novelty_routine(
            inds2update=ref_pop_inds,
            archive=archive,
            **fixed_attr_dict
        )

        added_inds, scs_inds_generated = outcome_archive.update(inv_pop)

        progression_monitoring.update(pop=inv_pop, outcome_archive=outcome_archive)

        if inv_pop.get_n_successful_inds() > 0:
            n_gen_rolling_reinit_research = 0

        # ---------------------------------------- FILL ARCHIVE --------------------------------------

        novelties_off = novelty_updated_inds[pop_size:] if algo_variant in consts.POP_BASED_ALGO_VARIANTS \
            else novelty_updated_inds

        archive.fill(
            pop2add=off,
            novelties=novelties_off
        )

        # ------------------------------------------ REPLACE -----------------------------------------

        replace_pop(
            pop=pop,
            ref_pop_inds=ref_pop_inds,
            **fixed_attr_dict
        )

        # -------------------------------------- MANAGE ARCHIVE --------------------------------------

        archive.manage_archive_size()

        # ----------------------------------------- MEASURE ------------------------------------------

        stats_tracker.update(
            pop=inv_pop, outcome_archive=outcome_archive, curr_n_evals=progression_monitoring.n_eval, gen=gen
        )

        do_dump_scs_archive = consts.DUMP_SCS_ARCHIVE_ON_THE_FLY and gen % consts.N_GEN_FREQ_DUMP_SCS_ARCHIVE == 0
        if do_dump_scs_archive:
            dump_archive_success_routine(
                timer=timer,
                timer_label=consts.NS_RUN_TIME_LABEL,
                run_name=run_name,
                curr_neval=progression_monitoring.n_eval,
                outcome_archive=outcome_archive,
            )

        curr_n_eval = progression_monitoring.n_eval
        if curr_n_eval > n_budget_rollouts:
            print(f'curr_n_eval={curr_n_eval} > n_budget_rollouts={n_budget_rollouts} | End of running.')
            stats_tracker.n_generations = gen
            do_iterate_gen = False
            continue  # end of the evolutionary process

        pass  # end of generation

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


def run_qd(**kwargs):

    #for k in kwargs:
    #    print(f'Unused given key: {k}')

    timer = Timer()
    timer.start(label=consts.NS_RUN_TIME_LABEL)

    creator, toolbox = initialize_deap_tools(
        genotype_len=kwargs['genotype_len'],
        pop_size=kwargs['pop_size'],
        parallelize=kwargs['parallelize'],
        evo_process=kwargs['evo_process'],
        algo_variant=kwargs['algo_variant'],
        n_genes_per_keypoint=kwargs['controller_info']['n_genes_per_keypoint']
    )

    evaluator = Evaluator(  # is it required ?
        evo_process=kwargs['evo_process'],
        eval_func=kwargs['evaluation_function'],
        toolbox=toolbox
    )

    stats_tracker = StatsTracker(algo_variant=kwargs['algo_variant'])

    if kwargs['algo_variant'] in consts.PYRIBS_QD_VARIANTS:
        return run_qd_pyribs(
                creator=creator,
                toolbox=toolbox,
                evaluator=evaluator,
                stats_tracker=stats_tracker,
                genotype_len=kwargs['genotype_len'],
                reinit_research_flg=kwargs['reinit_research_flg'],
                n_budget_rollouts=kwargs['n_budget_rollouts'],
                outcome_archive_kwargs=kwargs['outcome_archive_kwargs'],
                bound_genotype_thresh=kwargs['bound_genotype_thresh'],
                closer_genome_init_func=kwargs['closer_genome_init_func'],
                prob_cx=kwargs['prob_cx'],
                run_name=kwargs['run_name'],
                run_details=kwargs['run_details'],
                algo_variant=kwargs['algo_variant'],
                timer=timer,
        )
    elif kwargs['algo_variant'] in consts.SERENE_QD_VARIANTS:
        return run_qd_serene(
            run_name=kwargs['run_name'],
            n_budget_rollouts=kwargs['n_budget_rollouts'],
            genotype_len=kwargs['genotype_len'],
            toolbox=toolbox,
            reinit_research_flg=kwargs['reinit_research_flg'],
            bound_genotype_thresh=kwargs['bound_genotype_thresh'],
            closer_genome_init_func=kwargs['closer_genome_init_func'],
            prob_cx=kwargs['prob_cx'],
            pop_size=kwargs['pop_size'],
            evaluator=evaluator,
            outcome_archive_kwargs=kwargs['outcome_archive_kwargs'],
            stats_tracker=stats_tracker,
            algo_variant=kwargs['algo_variant'],
            archive_kwargs=kwargs['archive_kwargs'],
            is_novelty_required=kwargs['is_novelty_required'],
            evo_process=kwargs['evo_process'],
            bd_indexes=kwargs['bd_indexes'],
            bd_filters=kwargs['bd_filters'],
            novelty_metric=kwargs['novelty_metric'],
            bd_bounds=kwargs['bd_bounds'],
            nb_offsprings_to_generate=kwargs['nb_offsprings_to_generate'],
            mut_flg=kwargs['mut_flg'],
            robot_name=kwargs['robot_name'],
            obj_vertices_poses=kwargs['obj_vertices_poses'],
            stabilized_obj_pose=kwargs['stabilized_obj_pose'],
            timer=timer,
            run_details=kwargs['run_details']
        )
    else:
        return run_qd_local(
            creator=creator,
            toolbox=toolbox,
            evaluator=evaluator,
            stats_tracker=stats_tracker,
            timer=timer,
            **kwargs
        )
