
import evolutionary_process
import sys

from utils.args_processor import get_qd_algo_args, initialize_cpu_multicore_data
from utils.common_tools import wrapped_partial
from utils.io_run_data import export_dict_pickle
from utils.common_tools import get_new_run_name
import utils.constants as consts

from algorithms.evaluate import evaluate_grasp_ind, exception_handler_evaluate_grasp_ind, \
    generate_init_genome_operational_space


# The following variables must be initialized in the global scope to make them accessible for any scoop worker
ENV, EVAL_KWARGS, MULTICORE_SHARED_CFG = initialize_cpu_multicore_data()
QD_ALGO_ARGS = get_qd_algo_args(cfg=MULTICORE_SHARED_CFG)


def generate_init_genome_operational_space_routine(i_sample):
    return generate_init_genome_operational_space(
        i_sample=i_sample, eval_kwargs=EVAL_KWARGS, robot=MULTICORE_SHARED_CFG['robot']['name'], env=ENV
    )


def evaluate_grasp_ind_routine(individual, eval_kwargs, with_quality=False, n_reset_safecheck=2):

    raise_exceptions_flg = True
    try:
        behavior, fitness, info = evaluate_grasp_ind(
            individual=individual,
            env=ENV,
            robot=MULTICORE_SHARED_CFG['robot']['name'],
            eval_kwargs=eval_kwargs,
            with_quality=with_quality,
            n_reset_safecheck=n_reset_safecheck

        )
    except Exception as e:
        # Might be raised in some pathological cases due to simulator issue: wrong returned value, ...
        if raise_exceptions_flg:
            raise e
        behavior, fitness, info = exception_handler_evaluate_grasp_ind(
            individual=individual, eval_kwargs=eval_kwargs, with_quality=with_quality
        )

    return behavior, fitness, info


def eval_func_wrapper(eval_kwargs, with_quality=False):
    return wrapped_partial(evaluate_grasp_ind_routine, eval_kwargs=eval_kwargs, with_quality=with_quality)


def run_qd_routine(algo_args, n_trials_safeguard=False):

    if not n_trials_safeguard:
        archive_success = evolutionary_process.run_qd_wrapper(algo_args)
        return archive_success

    archive_success = None
    is_all_pop_invalid = True
    n_try_ns = 0

    while is_all_pop_invalid:
        archive_success = evolutionary_process.run_qd_wrapper(algo_args)

        n_try_ns += 1
        if n_try_ns >= consts.N_TRY_NS_MAX:
            raise Exception(f"The initial population failed {consts.N_TRY_NS_MAX} times")

        is_all_pop_invalid = True if archive_success is None else False  # all invalid = collision

    return archive_success


def main_loop():

    run_name = get_new_run_name(
        log_path=MULTICORE_SHARED_CFG['output']['log_path'], folder_name=MULTICORE_SHARED_CFG['output']['folder_name']
    )

    QD_ALGO_ARGS['run_name'] = run_name
    QD_ALGO_ARGS['evaluation_function'] = eval_func_wrapper(eval_kwargs=MULTICORE_SHARED_CFG['evaluate']['kwargs'])

    export_dict_pickle(run_name=run_name, dict2export=MULTICORE_SHARED_CFG, file_name='config')

    QD_ALGO_ARGS['closer_genome_init_func'] = generate_init_genome_operational_space_routine \
        if MULTICORE_SHARED_CFG['env']['kwargs']['closer_init_flg'] else None

    if MULTICORE_SHARED_CFG['algorithm'] == 'qdmbs_s':
        QD_ALGO_ARGS['evo_process'] = 'ns_rand_multi_bd'

    archive_success = run_qd_routine(algo_args=QD_ALGO_ARGS, n_trials_safeguard=consts.N_TRIALS_SAFEGUARD)

    if len(archive_success) == 0:
        print("Empty success archive.")
        return consts.RETURN_SUCCESS_CODE

    print(f"End of running. Size of output success archive : {len(archive_success)}")
    print("Success.")

    return consts.RETURN_SUCCESS_CODE


if __name__ == "__main__":
    sys.exit(main_loop())  # Note: scoop always exits with traceback


