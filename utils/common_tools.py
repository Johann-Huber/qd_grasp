
import pdb

from functools import partial, update_wrapper
import numpy as np
from pathlib import Path

import utils.constants as consts


def wrapped_partial(func, *args, **kwargs):
    """Taken from: http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/"""
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def check_exit():
    """Debug func."""
    print('\nok')
    exit()


def arg_clean_str(x):
    return str(x).strip().lower()


def bound(behavior, bound_behavior):
    for i in range(len(behavior)):
        if behavior[i] < bound_behavior[i][0]:
            behavior[i] = bound_behavior[i][0]
        if behavior[i] > bound_behavior[i][1]:
            behavior[i] = bound_behavior[i][1]


def normalize(behavior, bound_behavior):
    for i in range(len(behavior)):
        range_of_interval = bound_behavior[i][1] - bound_behavior[i][0]
        mean_of_interval = (bound_behavior[i][0] + bound_behavior[i][1]) / 2
        behavior[i] = (behavior[i] - mean_of_interval) / (range_of_interval / 2)


def list_l2_norm(list1, list2):
    if len(list1) != len(list2):
        raise NameError('The two lists have different length')
    else:
        dist = 0
        for i in range(len(list1)):
            dist += (list1[i] - list2[i]) ** 2
        dist = dist ** (1 / 2)
        return dist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def get_local_run_name(log_path, folder_name):
    Path(log_path).mkdir(exist_ok=True)
    id_run_export, valid_run_name_is_found = 0, False
    while not valid_run_name_is_found:
        run_name = Path(f"{log_path}/{folder_name}{id_run_export}")
        if not run_name.exists():
            valid_run_name_is_found = True
        else:
            id_run_export += 1

    return run_name


def get_new_run_name(log_path, folder_name, verbose=True):

    run_name = get_local_run_name(log_path, folder_name)

    run_name.mkdir(exist_ok=True)

    if verbose:
        print(f'Output folder (run_name={run_name}) has been successfully build.')

    return run_name


def is_export_path_type_valid(dump_path, attempted_export_str='data'):
    if type(dump_path) not in consts.SUPPORTED_DUMP_PATH_TYPES:
        print(
            f'[plot_export_routine] Warning: dump_path not in {consts.SUPPORTED_DUMP_PATH_TYPES} '
            f'(type={type(dump_path)}; cannot export {attempted_export_str}.'
        )
        return False

    return True


def get_export_path_root(dump_path):
    assert type(dump_path) in consts.SUPPORTED_DUMP_PATH_TYPES
    return str(dump_path) if not isinstance(dump_path, str) else dump_path
