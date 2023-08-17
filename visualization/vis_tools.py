import pdb
from pathlib import Path
import glob
import gym
import yaml
import numpy as np
from utils.io_run_data import load_dict_pickle # fail due to folder tree


N_SCS_RUN_DATA_KEY = 'Number of successful individuals (archive_success len)'
CONFIG_PKL_FILE_NAME = 'config'
SCS_ARCHIVE_INDS_KEY = 'inds'
IS_SCS_ARCHIVE_INDS_KEY = 'is_scs_inds'
SCS_ARCHIVE_BEHAVIOR_KEY = 'behavior_descriptors'
SCS_ARCHIVE_FITNESS_KEY = 'fitnesses'

def get_last_dump_ind_file(folder):
    ind_root_path = folder/"success_archives"
    inds_folders = [path for path in glob.glob(f'{ind_root_path}/individuals_*.npz')]
    np_ext_str = '.npz'
    n_char2skip = len(np_ext_str)
    all_ids = [int(ind_f.split("/")[-1].split("_")[-1][:-n_char2skip]) for ind_f in inds_folders]
    id_last_dump = np.argmax(all_ids)
    last_dump_ind_file = inds_folders[id_last_dump]
    return last_dump_ind_file



def get_specific_dump_ind_file(folder, id):
    ind_root_path = folder/"success_archives"
    inds_folders = [path for path in glob.glob(f'{ind_root_path}/individuals_*.npz')]
    np_ext_str = '.npz'
    n_char2skip = len(np_ext_str)
    all_ids = [int(ind_f.split("/")[-1].split("_")[-1][:-n_char2skip]) for ind_f in inds_folders]
    if id not in all_ids:
        id_last_dump = np.argmax(all_ids)
        last_dump_ind_file = inds_folders[id_last_dump]
        queried_dump_ind_file = last_dump_ind_file
        print(f'id={id} not in inds for folder={folder}, taking last value = {id_last_dump}')
    else:
        id_in_ind_list = all_ids.index(id)
        queried_dump_ind_file = inds_folders[id_in_ind_list]
        print('found')
    return queried_dump_ind_file


def get_last_dump_ind_file_qd(folder):
    ind_root_path = folder/"success_archives_qd"
    inds_folders = [path for path in glob.glob(f'{ind_root_path}/individuals_*.npz')]
    np_ext_str = '.npz'
    n_char2skip = len(np_ext_str)
    all_ids = [int(ind_f.split("/")[-1].split("_")[-1][:-n_char2skip]) for ind_f in inds_folders]
    id_last_dump = np.argmax(all_ids)
    last_dump_ind_file = inds_folders[id_last_dump]
    return last_dump_ind_file


def get_folder_path_from_str(run_folder_path):
	folder_path = Path(run_folder_path)
	if len(list(folder_path.glob("**/run_details*.yaml"))) == 0:
		raise FileNotFoundError("No run found")
	return folder_path


def get_running_folders(directory):
    return [path.parent for path in Path(directory).glob("**/run_details*.yaml")]


def load_run_output_files(folder):

    run_details_path = next(folder.glob("**/run_details*.yaml"))
    run_infos_path = next(folder.glob("**/run_infos*.yaml"))

    with open(run_details_path, "r") as f:
        run_details = yaml.safe_load(f)

    with open(run_infos_path, "r") as f:
        run_infos = yaml.safe_load(f)

    with open(folder / "config.pkl", "r") as f:
        path2file = folder / f'{CONFIG_PKL_FILE_NAME}.pkl'
        cfg = load_dict_pickle(file_path=path2file)

    return run_details, run_infos, cfg


def is_there_success(run_infos):
    return run_infos[N_SCS_RUN_DATA_KEY] > 0



def load_all_inds(folder, individuals_id=None):

    if individuals_id is not None:
        ind_file = get_specific_dump_ind_file(folder, id=individuals_id)
    else:
        ind_file = get_last_dump_ind_file(folder)
    return np.load(ind_file)[SCS_ARCHIVE_INDS_KEY]


def load_all_inds_qd(folder, keep_only_success=False):
    ind_file = get_last_dump_ind_file_qd(folder)

    loaded_inds = np.load(ind_file)[SCS_ARCHIVE_INDS_KEY]
    if keep_only_success:
        is_scs_inds = np.load(ind_file)[IS_SCS_ARCHIVE_INDS_KEY]
        assert len(loaded_inds) == len(is_scs_inds)
        loaded_inds = loaded_inds[is_scs_inds]

    return loaded_inds


def load_all_behavior_descriptors(folder):
    ind_file = get_last_dump_ind_file(folder)
    return np.load(ind_file, allow_pickle=True)[SCS_ARCHIVE_BEHAVIOR_KEY]


def load_all_fitnesses(folder, individuals_id=None):

    if individuals_id is not None:
        ind_file = get_specific_dump_ind_file(folder, id=individuals_id)
    else:
        ind_file = get_last_dump_ind_file(folder)

    #pdb.set_trace()
    return np.load(ind_file, allow_pickle=True)[SCS_ARCHIVE_FITNESS_KEY]


def get_controller_data(cfg):
    controller_class = cfg['evaluate']['kwargs']['controller_class']
    controller_info = cfg['evaluate']['kwargs']['controller_info']
    return controller_class, controller_info


def init_env(cfg, display=False):
    env_kwargs = cfg['env']['kwargs']
    env_kwargs['initial_state'] = cfg['env']['initial_state']
    #print('init_state=', env_kwargs['initial_state'])
    env_kwargs['display'] = display

    return gym.make(**env_kwargs)




