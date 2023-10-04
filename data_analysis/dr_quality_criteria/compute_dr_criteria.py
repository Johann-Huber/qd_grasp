
import os.path

import sys
import time
from pathlib import Path
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial

from utils.io_run_data import export_dict_pickle
import utils.constants as consts
from algorithms.evaluate import init_controller
from visualization.vis_tools import load_run_output_files, init_env, load_all_inds, \
	get_folder_path_from_str
import gym_envs.envs.src.env_constants as env_consts
from algorithms.evaluate import InteractionMeasures
from data_analysis.dr_quality_criteria.dr_criteria_functions import get_all_quality_criteria

DEBUG = False  # True # False
FEW_INDS = False
DISPLAY_FLG = False  # True # False

N_STEPS_VERIFICATION = 15
SMOOTH_VIS_TIME_SLEEP_IN_SEC = 0.001 #0.02


def set_camera_replay_trajs(env):
	env.reset_camera_pose_for_specific_table.resetDebugVisualizerCamera(
		cameraDistance=0.9, cameraYaw=0, cameraPitch=200, cameraTargetPosition=[0, 0, 0.1]
	)


def export_all_quality_criteria(all_quality_criteria, run_folder_path):
	root_path = run_folder_path if run_folder_path[-1] == '/' else run_folder_path + '/'
	export_path = root_path + 'dr_qualities'
	if not os.path.isdir(export_path):
		os.mkdir(export_path)
	export_dict_pickle(
		run_name=export_path, dict2export=all_quality_criteria, file_name='computed_quality_criteria'
	)


def compute_quality_criteria(ind, env, eval_kwargs, robot_name):

	controller_class = eval_kwargs['controller_class']
	controller_info = eval_kwargs['controller_info']
	episode_length = int(controller_info["nb_iter"])

	env.reset(force_state_load=True)

	if DISPLAY_FLG:
		env.reset_camera_pose_for_specific_table(table_label=env_consts.sim_robot2table[robot_name])

	controller = init_controller(
		individual=ind,
		controller_class=controller_class,
		controller_info=controller_info,
		env=env
	)

	im = InteractionMeasures()

	nrmlized_pos_arm = env.get_joint_state(normalized=True)
	nrmlized_pos_arm_prev = nrmlized_pos_arm
	reward_cumulated = 0
	i_step = 0

	while i_step < episode_length:
		action = controller.get_action(i_step=i_step, nrmlized_pos_arm=nrmlized_pos_arm, env=env)

		_, reward, _, info = env.step(action)

		nrmlized_pos_arm = env.get_joint_state(normalized=True)

		if controller.grip_time is not None:
			if i_step >= controller.grip_time and not im.is_already_grasped:
				im.is_already_grasped = True
				im.curr_contact_object_table = info["contact object table"]
				im.i_start_closing = i_step

		if info['touch'] and not im.is_already_touched:
			# First touch of object
			im.pos_touch_time = info['end effector position']
			im.is_already_touched = True

			# Close grip at first touch
			controller.update_grip_time(grip_time=i_step)
			controller.last_i = i_step
			nrmlized_pos_arm = nrmlized_pos_arm_prev

		if info['touch'] and im.i_start_closing is not None:
			im.t_touch_t_close_diff = i_step - im.i_start_closing
			im.i_start_closing = None

		if consts.AUTO_COLLIDE and info['autocollision']:
			return None

		reward_cumulated += reward

		is_robot_touching_table = len(info['contact robot table']) > 0
		im.robot_has_touched_table = im.robot_has_touched_table or is_robot_touching_table

		grasped_while_closing = im.t_touch_t_close_diff < eval_kwargs[
			'n_it_closing_grip'] * consts.GRASP_WHILE_CLOSE_TOLERANCE
		obj_not_touching_table = len(im.curr_contact_object_table) > 0
		is_there_grasp = reward and grasped_while_closing and obj_not_touching_table
		is_first_grasp = is_there_grasp and im.n_steps_before_grasp == consts.INF_FLOAT_CONST
		if is_first_grasp:
			im.n_steps_before_grasp = i_step

		if env.display:
			time.sleep(consts.TIME_SLEEP_SMOOTH_DISPLAY_IN_SEC)

		nrmlized_pos_arm_prev = nrmlized_pos_arm

		i_step += 1

	grasped_while_closing = im.t_touch_t_close_diff < eval_kwargs['n_it_closing_grip'] * consts.GRASP_WHILE_CLOSE_TOLERANCE
	obj_not_touching_table = len(im.curr_contact_object_table) > 0
	is_there_grasp = reward and grasped_while_closing and obj_not_touching_table

	return get_all_quality_criteria(
		interaction_measures=im,
		is_there_grasp=is_there_grasp,
		reward_cumulated=reward_cumulated,
		env=env,
		controller=controller,
		individual=ind,
		**eval_kwargs)


def launch_quality_criteria_extraction(run_folder_path):

	print(f'Computation of quality criteria.')

	# Initialization
	folder = get_folder_path_from_str(run_folder_path)
	run_details, run_infos, cfg = load_run_output_files(folder)

	robot_name = cfg['robot']['name']
	eval_kwargs = cfg['evaluate']['kwargs']

	env = init_env(cfg, display=DISPLAY_FLG)

	individuals = load_all_inds(folder)

	if len(individuals) == 0:
		print(f'folder={folder} no individuals to display. Ending.')
		return

	print(f'{len(individuals)} individuals have been found.')

	extract_qual_crit_wrapper = partial(
		compute_quality_criteria,
		env=env,
		eval_kwargs=eval_kwargs,
		robot_name=robot_name
	)

	if FEW_INDS:
		individuals = individuals[:5]

	if DEBUG:
		individuals = individuals[:5]
		all_quality_criteria = []

		for ind in individuals:
			quality_crit = extract_qual_crit_wrapper(ind)
			all_quality_criteria.append(quality_crit)

	else:
		with Pool() as p:
			all_quality_criteria = np.array(p.map(extract_qual_crit_wrapper, individuals))

	all_quality_criteria = {
		i_ind: {'individual': ind, 'quality_criteria': qc}
		for i_ind, (ind, qc) in enumerate(zip(individuals, all_quality_criteria))
	}
	print('done.')

	# Export
	export_all_quality_criteria(all_quality_criteria=all_quality_criteria, run_folder_path=run_folder_path)

	print('Ending...')
	env.close()


def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--runs", help="The directory containing runs", type=str,
						default=str(Path(__file__).parent.parent / 'runs'))
	return parser.parse_args()


def get_extract_qc_kwargs():
	args = arg_parser()
	return {'run_folder_path': args.runs}


def main():
	qc_kwargs = get_extract_qc_kwargs()

	launch_quality_criteria_extraction(**qc_kwargs)


if __name__ == "__main__":
	sys.exit(main())
