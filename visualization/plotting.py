import sys

import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
#import joypy
import glob
import json
import seaborn as sns
import argparse
from matplotlib import cm
import re
import os
import pdb
import pickle
import math
import yaml
from yaml.loader import SafeLoader

import utils.common_tools as uct

import utils.constants as consts

def plotting_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-path",
                        type=str,
                        default=None,
                        help="Path to the dump folder which contains runnning data to plot.")
    args = parser.parse_args()
    return args


def arg_parse_dump_path():
    args = plotting_arg_parser()
    dump_path = args.run_path

    if dump_path is None:
        raise AttributeError('Unset dump_path. (Call as : plotting.py --dump-path [path]')

    # Makes sure the dump path follows the right pattern
    dump_path = dump_path if dump_path[-1] != '/' else dump_path[:-1]

    return dump_path


def load_output_data(dump_path):

    run_name2load = dump_path
    details_export_pkl = run_name2load + '/details_export.pkl'
    data_export_pkl = run_name2load + '/data_export.pkl'

    with open(details_export_pkl, 'rb') as f:
        details_dict = pickle.load(f)

    with open(data_export_pkl, 'rb') as f:
        data_dict = pickle.load(f)

    return details_dict, data_dict


def load_run_info(dump_path):
    run_name2load = dump_path
    run_info_path = run_name2load + '/run_infos.yaml'

    # Open the file and load the file
    with open(run_info_path) as f:
        run_info_data = yaml.load(f, Loader=SafeLoader)

    return run_info_data


def path_safe_check(path):
    if not os.path.exists(path):
        raise AttributeError(f'Path does not exists : {path}')

def export_path_safe_create(path):
    if not os.path.exists(path):
        print(f'path={path} does not exists : creating it.')
        os.makedirs(path)
        print(f'path={path} successfully created.')
    else:
        print(f'path={path} already exists. Exporting path might overwrite previously generated plots.')



def plot_outcome_archive_coverage(details_dict, data_dict, title='', export_path=None):

    outcome_archive_cvg_hist = data_dict['outcome_archive_cvg_hist']
    n_eval_hist = data_dict['n_evals_hist']

    assert len(outcome_archive_cvg_hist) == len(n_eval_hist)

    df = pd.DataFrame({'number of rollouts': n_eval_hist,
                       'output archive coverage': outcome_archive_cvg_hist})

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    sns.lineplot(x="number of rollouts", y='output archive coverage', data=df, ax=ax)

    title_str = title  #' '.join([title, 'per robots', str(rob)])
    plt.suptitle(title_str)
    fig.subplots_adjust(bottom=0.15, wspace=0.4, hspace=0.3, right=0.94, left=0.06, top=0.90)

    if export_path is not None:
        export_path = export_path + '/' if export_path[-1] != '/' else export_path
        fig_path = export_path + title_str.replace(' ', '_').replace('|', '_') + '.png'
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()
        plt.close()


def plot_success_archive_coverage(details_dict, data_dict, title='', export_path=None):
    success_archive_cvg_hist = data_dict['success_archive_cvg_hist']
    n_eval_hist = data_dict['n_evals_hist']

    assert len(success_archive_cvg_hist) == len(n_eval_hist)

    df = pd.DataFrame({'number of rollouts': n_eval_hist,
                       'success archive coverage': success_archive_cvg_hist})

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    sns.lineplot(x="number of rollouts", y='success archive coverage', data=df, ax=ax)

    title_str = title  # ' '.join([title, 'per robots', str(rob)])
    plt.suptitle(title_str)
    fig.subplots_adjust(bottom=0.15, wspace=0.4, hspace=0.3, right=0.94, left=0.06, top=0.90)

    if export_path is not None:
        export_path = export_path + '/' if export_path[-1] != '/' else export_path
        fig_path = export_path + title_str.replace(' ', '_').replace('|', '_') + '.png'
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()
        plt.close()


def plot_success_archive_qd_score(details_dict, data_dict, title='', export_path=None):
    success_archive_qd_score_hist = data_dict['success_archive_qd_score_hist']
    n_eval_hist = data_dict['n_evals_hist']

    assert len(success_archive_qd_score_hist) == len(n_eval_hist)

    df = pd.DataFrame({'number of rollouts': n_eval_hist,
                       'success archive qd score': success_archive_qd_score_hist})

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    sns.lineplot(x="number of rollouts", y='success archive qd score', data=df, ax=ax)

    title_str = title  # ' '.join([title, 'per robots', str(rob)])
    plt.suptitle(title_str)
    fig.subplots_adjust(bottom=0.15, wspace=0.4, hspace=0.3, right=0.94, left=0.06, top=0.90)

    if export_path is not None:
        export_path = export_path + '/' if export_path[-1] != '/' else export_path
        fig_path = export_path + title_str.replace(' ', '_').replace('|', '_') + '.png'
        plt.savefig(fig_path)
        print(f"{fig_path} has been successfully exported.")
    else:
        plt.show()
        plt.close()



def plot_export_routine(dump_path, details_dict, data_dict):

    # Dump path sanity checks

    if not uct.is_export_path_type_valid(dump_path, attempted_export_str='plots'):
        return

    export_path_root = uct.get_export_path_root(dump_path=dump_path)
    export_path = export_path_root + '/plots'
    export_path_safe_create(export_path)


    plot_outcome_archive_coverage(details_dict, data_dict, export_path=export_path,
                                  title='outcome archive coverage over n evals')

    plot_success_archive_coverage(details_dict, data_dict, export_path=export_path,
                                  title='success archive coverage over n evals')

    plot_success_archive_qd_score(details_dict, data_dict, export_path=export_path,
                                  title='success archive qd score over n evals')

    plt.close('all')


def local_plotting(dump_path):
    """Run the script locally (not used as an external module)."""
    path_safe_check(dump_path)

    details_dict, data_dict = load_output_data(dump_path)
    plot_export_routine(dump_path=dump_path, details_dict=details_dict, data_dict=data_dict)

    print('local_plotting() : running over.')


def main_local_plot():
    dump_path = arg_parse_dump_path()
    local_plotting(dump_path)


if __name__ == "__main__":
    sys.exit(main_local_plot())


