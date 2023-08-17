
import evolutionary_process
import sys

from utils.args_processor import get_qd_algo_args, initialize_cpu_multicore_data
from utils.common_tools import wrapped_partial
from utils.io_run_data import export_dict_pickle
from utils.common_tools import get_new_run_name
import utils.constants as consts

from algorithms.evaluate import evaluate_grasp_ind, exception_handler_evaluate_grasp_ind


# The following variables must be initialized in the global scope to make them accessible for any scoop worker
ENV, EVAL_KWARGS, MULTICORE_SHARED_CFG = initialize_cpu_multicore_data()
QD_ALGO_ARGS = get_qd_algo_args(cfg=MULTICORE_SHARED_CFG)


def evaluate_grasp_ind_routine(individual, eval_kwargs):

    raise_exceptions_flg = True
    try:
        behavior, fitness, info = evaluate_grasp_ind(
            individual=individual,
            env=ENV,
            eval_kwargs=eval_kwargs
        )
    except Exception as e:
        # Might be raised in some pathological cases due to simulator issues
        if raise_exceptions_flg:
            raise e
        behavior, fitness, info = exception_handler_evaluate_grasp_ind(
            individual=individual, eval_kwargs=eval_kwargs,
        )

    return behavior, fitness, info


def eval_func_wrapper(eval_kwargs):
    """Wrap the eval function with predefied args so that it only require an individual as input
    (i.e. output = wrapped_eval(ind))"""
    return wrapped_partial(evaluate_grasp_ind_routine, eval_kwargs=eval_kwargs)


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


def init_run_dump_folder():
    """Must only be called in the main thread."""
    run_name = get_new_run_name(
        log_path=MULTICORE_SHARED_CFG['output']['log_path'], folder_name=MULTICORE_SHARED_CFG['output']['folder_name']
    )

    QD_ALGO_ARGS['run_name'] = run_name


def init_wrapped_eval_func():
    """Must only be called in the main thread."""
    QD_ALGO_ARGS['evaluation_function'] = eval_func_wrapper(eval_kwargs=MULTICORE_SHARED_CFG['evaluate']['kwargs'])


def dump_input_arguments():
    """Dump arguments that will be given to the QD algorithm to generate grasping trajectories."""
    export_dict_pickle(run_name=QD_ALGO_ARGS['run_name'], dict2export=MULTICORE_SHARED_CFG, file_name='config')


def end_of_run_routine(archive_success):
    """Must only be called in the main thread."""
    if len(archive_success) == 0:
        print("Empty success archive.")
        return consts.RETURN_SUCCESS_CODE

    print(f"End of running. Size of output success archive : {len(archive_success)}")
    print("Success.")

    return consts.RETURN_SUCCESS_CODE


def main_loop():
    """QD-Grasp entry point."""

    # Initialize main thread's arguments
    init_run_dump_folder()
    init_wrapped_eval_func()

    # Locally save params
    dump_input_arguments()

    # QD algorithm execution
    archive_success = run_qd_routine(algo_args=QD_ALGO_ARGS, n_trials_safeguard=consts.N_TRIALS_SAFEGUARD)

    # End of running
    return end_of_run_routine(archive_success)


if __name__ == "__main__":
    sys.exit(main_loop())  # Note: scoop always exits with traceback


