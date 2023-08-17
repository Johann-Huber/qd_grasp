#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pdb

import sys
import time
from pathlib import Path
import argparse
import numpy as np
import random
from dataclasses import dataclass, field

import utils.constants as consts
from algorithms.evaluate import init_controller, stabilize_simulation
from visualization.vis_tools import load_run_output_files,get_controller_data,init_env, load_all_inds, \
	get_folder_path_from_str, load_all_fitnesses


DISPLAY_FLG = True

FORCE_END_SUCCESS = True
N_STEPS_VERIFICATION = 15
SMOOTH_VIS_TIME_SLEEP_IN_SEC = 0.02


def set_camera_replay_trajs(env):
	env.p.resetDebugVisualizerCamera(
		cameraDistance=0.9, cameraYaw=0, cameraPitch=200, cameraTargetPosition=[0, 0, 0.1]
	)


@dataclass
class InteractionMeasures:
	is_already_touched: bool = False
	robot_has_touched_table: bool = False

	is_already_grasped: bool = False

	i_start_closing: int = None
	pos_touch_time: bool = None

	n_steps_before_grasp: int = consts.INF_FLOAT_CONST
	t_touch_t_close_diff: float = consts.INF_FLOAT_CONST
	curr_contact_object_table: list = field(default_factory=list)


def display_grasping_trajectory(ind, env, controller_class, controller_info):

	env.reset(load='state')
	stabilize_simulation(env=env)

	episode_length = int(controller_info["nb_iter"])
	nrmlized_pos_arm = env.get_joint_state(normalized=True)
	nrmlized_pos_arm_prev = nrmlized_pos_arm

	set_camera_replay_trajs(env)

	controller = init_controller(
		individual=ind,
		controller_class=controller_class,
		controller_info=controller_info,
		env=env
	)

	im = InteractionMeasures()
	is_scs = False
	it_delay_episode_end_from_closure_start = controller_info['n_it_closing_grip'] + N_STEPS_VERIFICATION
	i_step_end_episode = None

	for i_step in range(episode_length):

		if im.is_already_touched:
			if i_step > i_step_end_episode and FORCE_END_SUCCESS:
				break

		action = controller.get_action(i_step=i_step, nrmlized_pos_arm=nrmlized_pos_arm, env=env)
		_, reward, _, info = env.step(action)

		nrmlized_pos_arm = env.get_joint_state(normalized=True)

		if info['touch'] and not im.is_already_touched:
			im.is_already_touched = True

			# close gripper at first touch
			controller.update_grip_time(grip_time=i_step)
			controller.last_i = i_step
			nrmlized_pos_arm = nrmlized_pos_arm_prev
			i_step_end_episode = i_step + it_delay_episode_end_from_closure_start

		if info['is_success']:
			is_scs = True

		im.robot_has_touched_table = im.robot_has_touched_table or len(info['contact robot table']) > 0

		time.sleep(SMOOTH_VIS_TIME_SLEEP_IN_SEC)

		nrmlized_pos_arm_prev = nrmlized_pos_arm

	return is_scs


def select_successful_inds_only(inds):
	scs_inds = []
	for ind in inds:
		if ind.info.values['is_success']:
			scs_inds.append(ind)
	return scs_inds


def replay_trajs(run_folder_path, n_max_inds_disp=None, i_ind2display=None):

	print(f'Displaying running output.')

	folder = get_folder_path_from_str(run_folder_path)
	run_details, run_infos, cfg = load_run_output_files(folder)

	controller_class, controller_info = get_controller_data(cfg)
	controller_info['env_name'] = cfg['robot']['kwargs']['env_name']

	env = init_env(cfg, display=DISPLAY_FLG)

	individuals = load_all_inds(folder)

	if len(individuals) == 0:
		print(f'folder={folder} no individuals to display. (Is -os triggered ? Is there success ?). Ending.')
		return

	print(f'{len(individuals)} individuals have been found.')

	if i_ind2display is not None:
		ind = individuals[i_ind2display]

		print(f'=' * 20)
		print(f'Displaying ind n°{i_ind2display}')
		n_loop_display = 10
		for _ in range(n_loop_display):
			is_scs = display_grasping_trajectory(
				ind=ind,
				env=env,
				controller_class=controller_class,
				controller_info=controller_info,
			)

			print(f'ind={i_ind2display} | is_scs={is_scs}')
		pdb.set_trace()

	n_found_inds = len(individuals)

	random_inds = False
	if random_inds:
		ids_individuals = [*range(n_found_inds)]
		random.shuffle(ids_individuals)
		individuals = individuals[ids_individuals]

	best_inds = True
	if best_inds:
		ind_fits = load_all_fitnesses(folder)
		id_ind_sorted_by_fit = np.argsort(ind_fits)[::-1]
		individuals = [individuals[id_ind] for id_ind in id_ind_sorted_by_fit]
		ids_individuals = id_ind_sorted_by_fit

	if n_max_inds_disp is not None:
		print(f'Displaying {n_max_inds_disp} randomly picked individuals.')
	individuals = individuals[:n_max_inds_disp] if n_max_inds_disp is not None else individuals

	all_success_ids = []

	for ind_id, ind in zip(ids_individuals, individuals):
		print(f'=' * 20)
		print(f'Displaying ind n°{ind_id}')

		if best_inds:
			print('fit = ', ind_fits[ind_id])

		is_scs = display_grasping_trajectory(
			ind=ind,
			env=env,
			controller_class=controller_class,
			controller_info=controller_info,
		)

		if is_scs:
			all_success_ids.append(ind_id)

		print(f'Displaying ind n°{ind_id} : done')

	print('all_success_ids=', all_success_ids)

	env.reset(load='state')
	input("press enter to close")
	env.close()


def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--runs", help="The directory containing runs", type=str,
						default=str(Path(__file__).parent.parent / 'runs'))
	parser.add_argument("-nd", "--n_max_disp", help="Max number of inds to display. Default=None : display all inds.",
						type=int, default=None)
	parser.add_argument("-i", "--i_ind", help="Index of a specific ind to display. Default=None : display all inds.",
						type=int, default=None)
	return parser.parse_args()


def get_replay_trajs_kwargs():
	args = arg_parser()

	return {
		'run_folder_path': args.runs,
		'n_max_inds_disp': args.n_max_disp,
		'i_ind2display': args.i_ind,
	}


def main():
	rt_kwargs = get_replay_trajs_kwargs()
	replay_trajs(**rt_kwargs)


if __name__ == "__main__":
	sys.exit(main())
