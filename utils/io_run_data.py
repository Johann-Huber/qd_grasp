import pdb

import numpy as np
import pickle
import ruamel.yaml
import json

from visualization.plotting import plot_export_routine

import utils.common_tools as uct


yaml = ruamel.yaml.YAML()
yaml.width = 10000  # this is the output line width after which wrapping occurs


def export_run_dicts(run_name, details_export, data_export, run_infos_export, is_qd=False):
    is_qd_str = '_qd' if is_qd else ''

    details_export_pkl = str(run_name) + f'/details_export{is_qd_str}.pkl'
    data_export_pkl = str(run_name) + f'/data_export{is_qd_str}.pkl'
    run_infos_export_pkl = str(run_name) + f'/run_infos_export{is_qd_str}.pkl'

    dict_pkl_pairs = [(details_export, details_export_pkl),
                      (data_export, data_export_pkl),
                      (run_infos_export, run_infos_export_pkl)]

    for dict2export, pkl_path in dict_pkl_pairs:
        with open(pkl_path, 'wb') as f:
            pickle.dump(dict2export, f)
            print(f'Dict {pkl_path} has been successfully exported.')


def export_details_and_infos_yaml(dump_path, details, infos, is_qd=False):

    if not uct.is_export_path_type_valid(dump_path, attempted_export_str='run details and info'):
        return

    is_qd_str = '_qd' if is_qd else ''

    export_path_root = uct.get_export_path_root(dump_path=dump_path)

    output_details_yaml = export_path_root + f'/run_details{is_qd_str}.yaml'
    output_run_infos_yaml = export_path_root + f'/run_infos{is_qd_str}.yaml'

    save_yaml(details, output_details_yaml)
    print(f'Dict {output_details_yaml} has been successfully exported.')
    save_yaml(infos, output_run_infos_yaml)
    print(f'Dict {output_run_infos_yaml} has been successfully exported.')



def export_running_data(dump_path, data):

    if not uct.is_export_path_type_valid(dump_path, attempted_export_str='running data'):
        return

    export_path_root = uct.get_export_path_root(dump_path=dump_path)
    output_data_npz = export_path_root + '/run_data'

    np.savez_compressed(output_data_npz, **data)
    print(f'Dict {output_data_npz}.npz has been successfully exported.')


def export_dict_pickle(run_name, dict2export, file_name):
    pkl_path = str(run_name) + f'/{file_name}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(dict2export, f)
        print(f'Dict {pkl_path} has been successfully exported.')

def load_dict_pickle(file_path):
    file = open(file_path, 'rb')
    loaded_dict = pickle.load(file)
    file.close()
    return loaded_dict


def export_dict_json(export_path_root, dict2export, file_name):
    file2export = export_path_root + '/' + file_name
    with open(file2export, 'w', encoding='utf-8') as f:
        json.dump(dict2export, f, ensure_ascii=False, indent=4)

    print(f'{file2export} successfully exported.')

def read_json_as_dict(json_file):
    f = open(json_file)
    data_dict = json.load(f)
    f.close()
    return data_dict


def dump_archive_success_routine(timer, timer_label, run_name, curr_neval, outcome_archive, is_last_call=False):

    if is_last_call:
        print('\nLast dump of success archive...')

    elapsed_time = timer.get_on_the_fly_time(label=timer_label)
    outcome_archive.export(
        run_name=run_name,
        curr_neval=curr_neval,
        elapsed_time=elapsed_time,
        only_scs=True
    )

    if is_last_call:
        print(f'\nLatest success archive has been successfully dumped to {run_name}')

def setFlowStyle(seq):  # represent seq in flow style in the yaml file
    if isinstance(seq, np.ndarray) and np.ndim(seq) == 0:
        return seq.item()
    s = ruamel.yaml.comments.CommentedSeq(seq)
    s.fa.set_flow_style()
    return s


def save_yaml(data: dict, path=None):
    for key, value in data.items():  # setup yaml
        if isinstance(value, (tuple, list, np.ndarray)):
            data[key] = setFlowStyle(value)
        elif isinstance(value, dict):
            save_yaml(value)
    if path is not None:
        with open(path, 'w') as f:
            yaml.dump(data, f)


def export_running_data_routine(stats_tracker, run_name, run_details, run_infos):
    data_dict = stats_tracker.get_output_data()

    export_run_dicts(
        run_name=run_name,
        details_export=run_details,
        data_export=data_dict,
        run_infos_export=run_infos
    )

    export_details_and_infos_yaml(
        dump_path=run_name,
        details=run_details,
        infos=run_infos
    )

    export_running_data(
        dump_path=run_name,
        data=data_dict
    )

    plot_export_routine(
        dump_path=run_name,
        details_dict=run_details,
        data_dict=data_dict
    )

