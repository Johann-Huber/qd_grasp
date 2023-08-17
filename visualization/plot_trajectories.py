
import pdb

from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
from functools import partial
import argparse
import random
import numpy as np
import sys

from visualization.vis_tools import load_run_output_files, get_controller_data, init_env, load_all_inds, \
    get_folder_path_from_str
from algorithms.evaluate import init_controller, stabilize_simulation


DEBUG = True

N_INDS_VIS_DEFAULT = 1000 if not DEBUG else 10

DISPLAY_FLG_VIS = True
N_ITER_STABILIZING_SIM_DISPLAY = 1000

FORCE_END_SUCCESS = True
N_IT_VALIDATION = 15

PADDED_EFFECTOR_POS = np.array([0., 0., 0])  # 3D cartesian pose

# Color expresses temporality
COLOR_MAX = 255
PURPLE_RGB_CODE = [127, 0, 255]
YELLOW_RGB_CODE = [255, 138, 23]
COLOR_FROM = np.array(PURPLE_RGB_CODE) / COLOR_MAX
COLOR_TO = np.array(YELLOW_RGB_CODE) / COLOR_MAX
LINE_WIDTH_TRAJ_DISPLAY = 3


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs",
                        help="The directory containing runs",
                        type=str,
                        default=str(Path(__file__).parent.parent / 'runs'))
    parser.add_argument("-da", "--disp_all",
                        help=f"Display all successful trajs. Default to False : disp {N_INDS_VIS_DEFAULT} trajs.",
                        action="store_true")
    parser.add_argument("-nst", "--no_shuffle_trajs",
                        help="Determinist plotting: disp the first generated trajectories. Default to False : Randomly "
                             "choose trajectories to display.",
                        action="store_true")
    parser.add_argument("-ae", "--all_episode",
                        help="Display grasping the full trajectories. "
                             "Default to False: display grasping trajectories until the object is touch.",
                        action="store_true")
    return parser.parse_args()


# Global variables for parallelization
# todo refactore: scoop is not used anymore - global var should be avoided now
args = arg_parser()
FOLDER_STR = args.runs
DISP_ALL = args.disp_all
SHUFFLE_TRAJ = not args.no_shuffle_trajs
UNTIL_TOUCH = not args.all_episode

FOLDER = get_folder_path_from_str(FOLDER_STR)

RUN_DETAILS, RUN_INFOS, CFG = load_run_output_files(FOLDER)
CONTROLLER_CLASS, CONTROLLER_INFO = get_controller_data(CFG)
ENV = init_env(CFG, display=False)
INDIVIDUALS = load_all_inds(FOLDER)
N_MAX_TRAJ_TO_PLOT = len(INDIVIDUALS) if DISP_ALL else N_INDS_VIS_DEFAULT
NB_ITER = CONTROLLER_INFO['nb_iter']

CONTROLLER_INFO['env_name'] = CFG['robot']['kwargs']['env_name']


def get_traj_from_ind_deployement(ind, period=1):

    ENV.reset(load='state', force_state_load=True)
    stabilize_simulation(env=ENV)
    nrmlized_pos_arm = ENV.get_joint_state(normalized=True)
    nrmlized_pos_arm_prev = nrmlized_pos_arm

    controller = init_controller(
        individual=ind,
        controller_class=CONTROLLER_CLASS,
        controller_info=CONTROLLER_INFO,
        env=ENV
    )

    is_obj_touched = False

    display_traj_len = np.ceil(NB_ITER / period).astype(int)
    # trajectory = [(x,y,z)_0, ..., (x,y,z)_T]
    trajectory = np.stack([PADDED_EFFECTOR_POS] * display_traj_len)
    trajectory[0] = ENV.info['end effector position']

    is_already_touched = False
    it_delay_episode_end_from_closure_start = CONTROLLER_INFO['n_it_closing_grip'] + N_IT_VALIDATION
    i_step_end_episode = None

    for i_step in range(NB_ITER):

        if is_already_touched:
            if i_step > i_step_end_episode and FORCE_END_SUCCESS:
                break

        action = controller.get_action(i_step=i_step, nrmlized_pos_arm=nrmlized_pos_arm, env=ENV)

        _, reward, _, info = ENV.step(action)

        nrmlized_pos_arm = ENV.get_joint_state(normalized=True)

        is_displayed_step = i_step % period == 0
        if is_displayed_step:
            trajectory[i_step // period] = info['end effector position']

        if info['touch'] and not is_already_touched:
            is_already_touched = True

            # close gripper at first touch
            controller.update_grip_time(grip_time=i_step)
            controller.last_i = i_step
            nrmlized_pos_arm = nrmlized_pos_arm_prev
            i_step_end_episode = i_step + it_delay_episode_end_from_closure_start

        if UNTIL_TOUCH and info['touch']:
            is_obj_touched = True
            break

    if not is_obj_touched:
        return []  # not reproducible traj: discard

    return trajectory


def print_triggered_traj_leng_mode():
    if UNTIL_TOUCH:
        print('UNTIL_TOUCH=True : Display grasping trajectories until the object is touched.')
    else:
        print('Display grasping trajectories throughout the whole episode.')


def update_curr_color(curr_color, step_color):
    curr_color += step_color
    if (curr_color > COLOR_TO).all():
        curr_color = COLOR_TO

    return curr_color


def add_effector_pose_to_overall_display(prev_pos, position, curr_color, env):
    env.p.addUserDebugLine(prev_pos, position, curr_color, parentObjectUniqueId=-1, lineWidth=LINE_WIDTH_TRAJ_DISPLAY)


def extract_trajectories(individuals, period):
    n_inds = len(individuals)
    print(f'Extracting {n_inds} ... ', end='')

    with Pool() as p:
        trajectories = np.array(p.map(partial(get_traj_from_ind_deployement, period=period), individuals))
    print('done.')

    n_test_traj = len(trajectories)
    trajectories = [traj for traj in trajectories if len(traj) > 0]
    n_valid_traj = len(trajectories)
    n_discarded_traj = n_test_traj - n_valid_traj

    print(f'Note: Some trajs might be discarded due to simulation non-determinism during extraction. (This should only '
          f'discard very fragile trajectories).'
          f'Number of discarded trajs : {n_discarded_traj}')

    return trajectories


def set_camera_plot_trajs():
    ENV.p.resetDebugVisualizerCamera(
        cameraDistance=0.9, cameraYaw=0, cameraPitch=200, cameraTargetPosition=[0, 0, 0.1]
    )


def display_end_effector_pose_trajectories(trajectories):
    ENV = init_env(CFG, display=DISPLAY_FLG_VIS)
    ENV.reset()

    stabilize_simulation(env=ENV, n_step_stabilizing=N_ITER_STABILIZING_SIM_DISPLAY)

    set_camera_plot_trajs()

    len_traj = len(trajectories[0]) / 2
    step_color = (COLOR_TO - COLOR_FROM) / len_traj

    for id_traj, trajectory in enumerate(tqdm(trajectories, desc='displaying trajectories')):

        assert len(trajectory) > 0
        prev_pos = trajectory[0]
        curr_color = COLOR_FROM.copy()

        for i, position in enumerate(trajectory[1:]):  # t lies in [1, ..., T]

            if UNTIL_TOUCH and (position == PADDED_EFFECTOR_POS).all():
                break

            curr_color = update_curr_color(curr_color, step_color)
            add_effector_pose_to_overall_display(prev_pos=prev_pos, position=position, curr_color=curr_color, env=ENV)
            prev_pos = position

    input(f"{len(trajectories)} trajectory successfully plotted. Press enter to close...")


def plot_trajectories(period=4):
    n_found_inds = len(INDIVIDUALS)
    n_inds_vis = N_INDS_VIS_DEFAULT if not DISP_ALL else n_found_inds

    if SHUFFLE_TRAJ:
        ids_individuals = [*range(n_found_inds)]
        random.shuffle(ids_individuals)
        individuals = INDIVIDUALS[ids_individuals][:n_inds_vis]
        print(f'{n_inds_vis} individuals have been randomly selected.')
    else:
        individuals = INDIVIDUALS[:n_inds_vis]
        print(f'The first {n_inds_vis} successful individuals have been selected.')

    if len(INDIVIDUALS) == 0:
        print(f'folder={FOLDER} no individual to plot. Ending.')
        return

    print_triggered_traj_leng_mode()

    trajectories = extract_trajectories(individuals=individuals, period=period)

    display_end_effector_pose_trajectories(trajectories)


def main():
    plot_trajectories()


if __name__ == '__main__':
    sys.exit(main())


