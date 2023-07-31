#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pdb
import sys
import time
from pathlib import Path
import argparse
import gym
import numpy as np
#import gym_envs
import random

import utils.constants as consts

from algorithms.evaluate import init_controller, stabilize_simulation

from visualization.vis_tools import load_run_output_files,get_controller_data,init_env, load_all_inds, \
	get_folder_path_from_str, load_all_fitnesses


DISPLAY_FLG = True


def get_max_speed_and_torque(env, info, max_speed, max_torque):
	states = env.p.getJointStates(env.robot_id, env.joint_ids[:-env.n_control_gripper])
	max_speed = np.max([max_speed, *[s[1] for s in states]])
	max_torque = np.max([max_torque, np.abs(info['applied joint motor torques']).max()])
	return max_speed, max_torque


def display_grasping_trajectory(ind, env, speed, nb_steps_to_rollout, controller_class, controller_info):

	env.reset(load='state')
	stabilize_simulation(env=env)
	nrmlized_pos_arm = env.get_joint_state(normalized=True)
	nrmlized_pos_arm_prev = nrmlized_pos_arm

	#env.camera.update({'target': [0, 0, 0], 'distance': 1, 'yaw': 180, 'pitch': -40},)
	env.p.resetDebugVisualizerCamera(
		cameraDistance=0.9, cameraYaw=0, cameraPitch=200, cameraTargetPosition=[0, 0, 0.1]
	)

	controller = init_controller(
		individual=ind,
		controller_class=controller_class,
		controller_info=controller_info,
		env=env
	)
	controller.update_grip_time(grip_time=consts.INF_FLOAT_CONST) # close while touch

	nb_iter = int(controller_info["nb_iter"])
	dt = env.p.getPhysicsEngineParameters()["fixedTimeStep"]
	max_speed, max_torque = 0, 0
	robot_has_touched_table = False
	is_scs = False
	is_already_touched = False

	last_time = time.perf_counter()


	n_it_verification = 15
	it_delay_episode_end_from_closure_start = controller_info['n_it_closing_grip'] + n_it_verification
	i_step_end_episode = None

	for i_step in range(nb_iter):

		if is_already_touched:
			if i_step > i_step_end_episode:
				print('force end : i_step=', i_step)
				print('nb_iter=', nb_iter)
				break # force episode end

		action = controller.get_action(i_step=i_step, nrmlized_pos_arm=nrmlized_pos_arm, env=env)
		_, reward, done, info = env.step(action)

		nrmlized_pos_arm = env.get_joint_state(normalized=True)

		if info['touch'] and not is_already_touched:
			is_already_touched = True

			# force closure at first touch
			controller.update_grip_time(grip_time=i_step)
			controller.last_i = i_step
			nrmlized_pos_arm = nrmlized_pos_arm_prev
			i_step_end_episode = i_step + it_delay_episode_end_from_closure_start


		if info['is_success']:
			is_scs = True
			pass

		max_speed, max_torque = get_max_speed_and_torque(env=env, info=info, max_speed=max_speed, max_torque=max_torque)
		#states = env.p.getJointStates(env.robot_id, env.joint_ids[:-env.n_control_gripper])
		#max_speed = np.max([max_speed, *[s[1] for s in states]])
		#max_torque = np.max([max_torque, np.abs(info['applied joint motor torques']).max()])
		robot_has_touched_table = robot_has_touched_table or len(info['contact robot table']) > 0

		# slow down for vizualization
		if speed is not None:
			now = time.perf_counter()
			#time.sleep(max(0, dt * nb_steps_to_rollout - (now - last_time)) + speed)
			time.sleep(0.05)
			#time.sleep(0.01)
			last_time = now
		else:
			#time.sleep(0.05)
			time.sleep(0.02)

		nrmlized_pos_arm_prev = nrmlized_pos_arm
		i_step += 1

	#pdb.set_trace()

	print("max_speed", max_speed, "max_torque", max_torque)

	return is_scs


def select_successful_inds_only(inds):
	scs_inds = []
	for ind in inds:
		if ind.info.values['is_success']:
			scs_inds.append(ind)
	return scs_inds


def replay_trajs(
	run_folder_path,
	speed: float = None,
	n_max_inds_disp: int = None,
	i_ind2display=None,
	debug: bool = False,
) -> None:

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

	nb_steps_to_rollout = cfg['evaluate']['kwargs']['nb_steps_to_rollout']

	if i_ind2display is not None:
		ind = individuals[i_ind2display]

		print(f'Displaying ind n°{i_ind2display} : done' + '=' * 20)
		n_loop_display = 10
		for _ in range(n_loop_display):
			is_scs = display_grasping_trajectory(
				ind=ind,
				env=env,
				speed=speed,
				controller_class=controller_class,
				controller_info=controller_info,
				nb_steps_to_rollout=nb_steps_to_rollout,
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

	#for ind_id, ind in enumerate(individuals):
	for ind_id, ind in zip(ids_individuals, individuals):
		print(f'Displaying ind n°{ind_id}' + '='*20)

		if best_inds:
			print('fit = ', ind_fits[ind_id])

		is_scs = display_grasping_trajectory(
			ind=ind,
			env=env,
			speed=speed,
			controller_class=controller_class,
			controller_info=controller_info,
			nb_steps_to_rollout=nb_steps_to_rollout,
		)

		if is_scs:
			all_success_ids.append(ind_id)

		print(f'Displaying ind n°{ind_id} : done' + '=' * 20)

	print('all_success_ids=', all_success_ids)

	env.reset(load='state')
	input("press enter to continue")
	env.close()


def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--runs", help="The directory containing runs", type=str,
						default=str(Path(__file__).parent.parent / 'runs'))
	parser.add_argument("-s", "--speed", help="Control the speed for visualization", type=float, default=None)  # high (ex 1., 10.) is slow, small (ex .1) is fast
	parser.add_argument("-nd", "--n_max_disp", help="Max number of inds to display. Default=None : display all inds.",
						type=int, default=None)
	parser.add_argument("-i", "--i_ind", help="Index of a specific ind to display. Default=None : display all inds.",
						type=int, default=None)
	parser.add_argument("-d", "--debug",
						action="store_true",
						help="Early dump debug. No safecheck for success")
	return parser.parse_args()


def main():
	args = arg_parser()

	run_folder_path = args.runs
	speed = args.speed
	n_max_inds_disp = args.n_max_disp
	i_ind2display = args.i_ind
	debug = args.debug

	replay_trajs(
		run_folder_path=run_folder_path,
		speed=speed,
		n_max_inds_disp=n_max_inds_disp,
		i_ind2display=i_ind2display,
		debug=debug,
	)


if __name__ == "__main__":
	sys.exit(main())
