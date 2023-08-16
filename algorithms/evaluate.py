

import numpy as np
import utils.constants as consts
import utils.common_tools as uct
import os

from utils.evo_tools import get_normalized_multi_fitness
import random
import gym_envs.envs.src.env_constants as env_consts
import pdb
import time
import algorithms.controllers.controller_params as ctrl_params


def evaluate_grasp_ind(individual, env, robot, eval_kwargs, n_reset_safecheck):

    evo_process = eval_kwargs['evo_process']
    bd_flg = eval_kwargs['bd_flg']
    add_iter = eval_kwargs['add_iter']
    nb_iter = eval_kwargs['nb_iter']
    bd_bounds = eval_kwargs['bd_bounds']
    #nb_steps_to_rollout = eval_kwargs['nb_steps_to_rollout']
    no_contact_table = eval_kwargs['no_contact_table']
    controller_class = eval_kwargs['controller_class']
    controller_info = eval_kwargs['controller_info']
    n_it_closing_grip = eval_kwargs['n_it_closing_grip']
    prehension_criteria = eval_kwargs['prehension_criteria']
    algo_variant = eval_kwargs['algo_variant']
    pdb.set_trace()
    env.reset(load='state' if consts.RESET_MODE else 'all')

    controller = init_controller(
        individual=individual,
        controller_class=controller_class,
        controller_info=controller_info,
        env=env
    )

    is_already_touched = False
    is_already_grasped = False
    robot_has_touched_table = False
    i_start_closing = None
    half_state_measured = False
    or_touch_time = None
    pos_touch_time = None
    half_time = nb_iter / 4
    bd_len = len(bd_bounds)
    init_obj_pos = env.info['object position']
    n_steps_before_grasp = consts.INF_FLOAT_CONST

    grip_info = {"contact object table": [], "diff(t_close, t_touch)": consts.INF_FLOAT_CONST}
    info_out = {'is_valid': None, 'energy': 0, 'n_steps_before_grasp': n_steps_before_grasp, 'reward': 0,
                'normalized_multi_fit': 0., 'touch_var': None,
                'is_obj_touched': False,
                }
    all_touch_on_obj, all_touch_on_robot = [], []
    is_discontinuous_touch, n_steps_continuous_touch, n_steps_discontinuous_touch = False, 0, 0
    max_observed_n_touch = 0

    all_pos_obj_before_grasp, all_orient_obj_before_grasp = [], []

    controller.update_grip_time(grip_time=consts.INF_FLOAT_CONST)  # disable grasping before touching

    nrmlized_pos_arm = env.get_joint_state(normalized=True)
    nrmlized_pos_arm_prev = nrmlized_pos_arm

    i_step = 0

    while i_step < nb_iter:
        action = controller.get_action(i_step=i_step, nrmlized_pos_arm=nrmlized_pos_arm, env=env)

        _, reward, done, info = env.step(action)

        nrmlized_pos_arm = env.get_joint_state(normalized=True)

        if done:
            pdb.set_trace()
            break

        if i_step >= controller.grip_time and not is_already_grasped:
            is_already_grasped = True
            grip_info["contact object table"] = info["contact object table"]
            i_start_closing = i_step  # the object should be grasped after having closed the gripper

        if not is_already_grasped:
            curr_obj_pos, curr_obj_orient = env.p.getBasePositionAndOrientation(env.obj_id)
            all_pos_obj_before_grasp.append(curr_obj_pos)
            all_orient_obj_before_grasp.append(curr_obj_orient)

        if info['touch'] and not is_already_touched:
            # first touch of object
            or_touch_time = info['end effector xyzw']
            pos_touch_time = info['end effector position']

            is_already_touched = True

            # Close grip at first touch
            controller.update_grip_time(grip_time=i_step)
            controller.last_i = i_step
            nrmlized_pos_arm = nrmlized_pos_arm_prev

        if info['touch'] and i_start_closing is not None:
            grip_info["diff(t_close, t_touch)"] = i_step - i_start_closing
            i_start_closing = None

        if i_step >= half_time and not half_state_measured:
            grip_or_half = np.array(info['end effector xyzw'])
            grip_pos_half = np.array(info['end effector position'])
            half_state_measured = True

        if consts.AUTO_COLLIDE and info['autocollision']:
            return get_autocollide_outputs(
                evo_process=evo_process,
                bd_len=bd_len,
                algo_variant=algo_variant,
                robot=robot,
                is_already_touched=is_already_touched
            )

        if is_already_touched and not is_discontinuous_touch:
            touch_points_on_obj = info['touch_points_obj']
            touch_points_on_robot = info['touch_points_robot']
            is_less_touching_obj = len(touch_points_on_obj) < max_observed_n_touch or \
                                   len(touch_points_on_robot) < max_observed_n_touch
            if is_less_touching_obj:
                is_discontinuous_touch = True
            else:
                n_steps_continuous_touch += 1
                max_observed_n_touch = max(max_observed_n_touch,
                                           len(touch_points_on_obj))
                all_touch_on_obj.append(touch_points_on_obj)
                all_touch_on_robot.append(touch_points_on_robot)

        if is_discontinuous_touch:
            n_steps_discontinuous_touch += 1

        info_out['reward'] += reward
        info_out['energy'] += np.abs(info['applied joint motor torques']).sum()
        is_robot_touching_table = len(info['contact robot table']) > 0
        robot_has_touched_table = robot_has_touched_table or is_robot_touching_table

        grasped_while_closing = grip_info[
                                    'diff(t_close, t_touch)'] < n_it_closing_grip * consts.GRASP_WHILE_CLOSE_TOLERANCE
        obj_not_touching_table = len(grip_info['contact object table']) > 0
        is_there_grasp = reward and grasped_while_closing and obj_not_touching_table
        is_first_grasp = is_there_grasp and n_steps_before_grasp == consts.INF_FLOAT_CONST
        if is_first_grasp:
            n_steps_before_grasp = i_step

        if env.display:
            time.sleep(consts.TIME_SLEEP_SMOOTH_DISPLAY_IN_SEC)

        nrmlized_pos_arm_prev = nrmlized_pos_arm

        i_step += 1

    # use last info to compute behavior and fitness
    grasped_while_closing = grip_info['diff(t_close, t_touch)'] < n_it_closing_grip * consts.GRASP_WHILE_CLOSE_TOLERANCE
    obj_not_touching_table = len(grip_info['contact object table']) > 0
    is_there_grasp = reward and grasped_while_closing and obj_not_touching_table

    if is_there_grasp:
        is_there_grasp = do_safechecks_traj_success(
            env=env, add_iter=add_iter, controller=controller, nb_iter=nb_iter, n_reset_safecheck=n_reset_safecheck,
            robot=robot
        )

    info_out['is_success'] = False
    if is_there_grasp:
        if or_touch_time is None:
            is_success = False
            info_out['is_success'] = False
            pos_touch_time = None
        else:
            if no_contact_table and robot_has_touched_table:
                is_success = False
                or_touch_time = None
            else:
                is_success = True
                info_out['is_success'] = True
    else:
        is_success = False
        or_touch_time = None

    info_out['is_success'] = is_success
    info_out['is_valid'] = True
    info_out['energy'] = (-1) * info_out['energy']  # minimize energy => maximise energy_fit = (-1) * energy

    if not is_already_touched:
        grip_or_half = None

    last_obj_pos = info['object position']

    behavior = build_behavior_vector(
        bd_flg=bd_flg,
        evo_process=evo_process,
        grip_pos_half=grip_pos_half,
        grip_or_half=grip_or_half,
        pos_touch_time=pos_touch_time,
        or_touch_time=or_touch_time,
        init_obj_pos=init_obj_pos,
        last_obj_pos=last_obj_pos,
        bd_bounds=bd_bounds
    )

    dummy_fitness = float(is_success)

    if is_success:
        info_out['touch_var'] = do_touch_variance_computation(
            all_touch_on_obj=all_touch_on_obj,
            all_touch_on_robot=all_touch_on_robot,
            n_steps_continuous_touch=n_steps_continuous_touch,
            max_observed_n_touch=max_observed_n_touch,
            n_steps_discontinuous_touch=n_steps_discontinuous_touch,
        ) if 'touch_var' in prehension_criteria else None
    else:
        info_out['touch_var'] = -consts.INF_FLOAT_CONST if 'touch_var' in prehension_criteria else None

    mono_eval_fit = info_out['touch_var']

    info_out['normalized_multi_fit'] = get_normalized_multi_fitness(
        energy_fit=info_out['energy'], mono_eval_fit=mono_eval_fit, robot_name=robot
    )
    info_out['is_obj_touched'] = is_already_touched

    if evo_process in consts.EVO_PROCESS_WITH_LOCAL_COMPETITION:
        dummy_fitness = (dummy_fitness, dummy_fitness)
    elif algo_variant in consts.ARCHIVE_BASED_NOVELTY_FITNESS_SELECTION_ALGO_VARIANTS:
        dummy_fitness = (dummy_fitness, dummy_fitness)
    else:
        dummy_fitness = (dummy_fitness,)

    return behavior.tolist(), dummy_fitness, info_out


def exception_handler_evaluate_grasp_ind(individual, eval_kwargs):

    evo_process = eval_kwargs['evo_process']
    prehension_criteria = eval_kwargs['prehension_criteria']

    # to do : plug it to an error logger

    print(f'DANGER : pathological case raised in eval. ind={individual}')
    n_expected_bds = (3 * 3 + 4 * 2)
    dummy_behavior_failure = n_expected_bds * [None]

    if evo_process in consts.EVO_PROCESS_WITH_LOCAL_COMPETITION:
        dummy_fitness_failure = (False, False)
    else:
        dummy_fitness_failure = (False,)

    dummy_info_out_failure = {
        'is_success': False,
        'is_valid': None,
        'energy': 0,
        'n_steps_before_grasp': consts.INF_FLOAT_CONST,
        'reward': 0,
        'normalized_multi_fit': 0.,
        'touch_var': -consts.INF_FLOAT_CONST if 'touch_var' in prehension_criteria else None,
    }
    return dummy_behavior_failure, dummy_fitness_failure, dummy_info_out_failure


def get_autocollide_outputs(evo_process, bd_len, algo_variant, robot, is_already_touched, verbose=True):

    if verbose:
        print('AUTOCOLLIDE !' + 20 * '=')

    worst_energy = env_consts.ENERGY_FIT_NORM_BOUNDS_PER_ROB[robot]['min']
    worst_fitness = -consts.INF_FLOAT_CONST
    worst_touch_var = -consts.INF_FLOAT_CONST
    worst_normalized_multi_fit = 0.

    # Build behavior descriptor
    behavior = build_invalid_behavior_vector(evo_process=evo_process, bd_len=bd_len)

    # Build fitness
    if evo_process in consts.EVO_PROCESS_WITH_LOCAL_COMPETITION:
        dummy_fitness = (worst_fitness, worst_fitness)
    elif algo_variant in consts.ARCHIVE_BASED_NOVELTY_FITNESS_SELECTION_ALGO_VARIANTS:
        dummy_fitness = (worst_fitness, worst_fitness)
    else:
        dummy_fitness = (worst_fitness,)

    # Build info_out
    info_out = {
        'is_valid': False,
        'is_success': False,
        'auto_collided': True,
        'energy': worst_energy,
        'is_obj_touched': is_already_touched,
        'normalized_multi_fit': worst_normalized_multi_fit,
        'touch_var': worst_touch_var,
    }

    return behavior, dummy_fitness, info_out


def around_inds(inds):
    n_around = consts.N_DIGITS_AROUND_INDS_EVAL
    inds = np.around(np.array(inds), n_around)

    return inds


def init_controller(individual, controller_class, controller_info, env):
    individual = around_inds(individual)
    controller_info['individual'] = individual

    controller_info['initial'] = env.get_joint_state(position=True)

    return controller_class(**controller_info)


def stabilize_simulation(env, n_step_stabilizing=None):
    n_iter_stab = consts.N_ITER_STABILIZING_SIM if n_step_stabilizing is None else n_step_stabilizing
    for i in range(n_iter_stab):
        #print('i=', i)
        env.p.stepSimulation()


def build_invalid_behavior_vector(evo_process, bd_len):
    return [None if evo_process == 'ns_rand_multi_bd' else 0] * bd_len


def fillna_behavior(behavior, evo_process):
    if evo_process not in consts.MULTI_BD_EVO_PROCESSES:
        behavior = np.where(behavior == None, consts.FIXED_VALUE_UNDEFINED_BD_FILL, behavior)
    return behavior


def build_behavior_vector(bd_flg, evo_process, grip_pos_half, grip_or_half, pos_touch_time, or_touch_time, init_obj_pos,
                          last_obj_pos, bd_bounds):

    bd_len = len(bd_bounds)
    behavior = np.array([None] * bd_len, dtype=object)

    if bd_flg == 'pos_touch':
        behavior[:] = pos_touch_time
        behavior = fillna_behavior(behavior=behavior, evo_process=evo_process)
        return behavior

    if bd_flg == 'last_pos_obj_pos_touch':
        last_obj_pos_offset = [
            last_obj_pos[0] - init_obj_pos[0],
            last_obj_pos[1] - init_obj_pos[1],
            last_obj_pos[2]
        ]
        behavior[:3] = last_obj_pos_offset
        uct.bound(behavior[:3], bd_bounds[:3])

        behavior[3:6] = pos_touch_time
        behavior = fillna_behavior(behavior=behavior, evo_process=evo_process)
        return behavior

    if bd_flg == 'last_pos_obj_pos_touch_pos_half':
        last_obj_pos_offset = [
            last_obj_pos[0] - init_obj_pos[0],
            last_obj_pos[1] - init_obj_pos[1],
            last_obj_pos[2]
        ]
        behavior[:3] = last_obj_pos_offset
        uct.bound(behavior[:3], bd_bounds[:3])

        behavior[3:6] = pos_touch_time
        behavior[6:9] = grip_pos_half

        behavior = fillna_behavior(behavior=behavior, evo_process=evo_process)
        return behavior

    behavior[:3] = [last_obj_pos[0] - init_obj_pos[0],
                    last_obj_pos[1] - init_obj_pos[1],
                    last_obj_pos[2]]

    uct.bound(behavior[:3], bd_bounds[:3])

    if or_touch_time is not None:
        or_touch_time = or_touch_time.elements  # Quaternion to numpy array

    behavior[3:7] = or_touch_time  # this one is common to the 3
    behavior[7:10] = pos_touch_time

    if bd_flg == 'nsmbs':
        behavior[10:] = grip_or_half
    elif bd_flg == 'all_bd':
        behavior[10:14] = grip_or_half
        behavior[14:] = grip_pos_half
    elif bd_flg == 'bd_safe_real':
        behavior[10:14] = or_touch_time
        behavior[14:] = pos_touch_time
    else:
        raise RuntimeError(f'Unknown bd_flg: {bd_flg}')

    behavior = fillna_behavior(behavior=behavior, evo_process=evo_process)

    return behavior


def do_env_stabilization_check(env, add_iter, controller):
    """Apply add_iter steps to env while maintaining end effector at its current position. Return true if the reward if
     non-null, i.e. if the object is still considered as grasped (stable grasp)."""
    if controller.get_grip_control_mode() == ctrl_params.GripControlMode.WITH_SYNERGIES:
        gpl = controller.grasp_primitive_label
        static_pos_action = np.hstack((env.get_joint_state(position=True, normalized=True), -1, gpl))
    else:
        static_pos_action = np.hstack((env.get_joint_state(position=True, normalized=True), -1))

    # static_pos_action = [ current arm position + closed gripper ]
    for i in range(add_iter):
        _, reward, _, _ = env.step(static_pos_action)
    is_there_grasp = reward

    return is_there_grasp


def do_traj_redeploiment_safecheck(
        env, controller, nb_iter, n_reset_safecheck, init_rand_pos=None, do_noise_joints_pos=False
):
    """Deplay controller's trajectory nb_iter times, to make sure it always produce a successful grasp.
     Returns True is all attempts are successful, False otherwise. """

    assert nb_iter > 0
    is_there_grasp = True

    for _ in range(n_reset_safecheck):
        env.reset(init_pos_offset=init_rand_pos, do_noise_joints_pos=do_noise_joints_pos)
        nrmlized_pos_arm = env.get_joint_state(normalized=True)
        #end_effector_pos = env.get_end_effector_state()
        controller.reset_rolling_attributes()

        for i_step in range(nb_iter):
            action = controller.get_action(i_step=i_step,
                                           nrmlized_pos_arm=nrmlized_pos_arm,
                                           env=env)
            _, reward, done, info = env.step(action)
            nrmlized_pos_arm = env.get_joint_state(normalized=True)

        if not info['is_success']:
            is_there_grasp = False
            break  #  fails : early stop

    return is_there_grasp


def do_safechecks_traj_success(robot, env, add_iter, controller, nb_iter, n_reset_safecheck):
    is_stable_grasp = do_env_stabilization_check(env=env, add_iter=add_iter, controller=controller)
    if not is_stable_grasp:
        return False

    skip_redeploiement_safecheck = robot in env_consts.ROBOT_TYPE_SKIP_REDEPLOIEMENT_SAFECHECK
    if skip_redeploiement_safecheck:
        return True

    is_grasp_reproducible = do_traj_redeploiment_safecheck(
        env=env, controller=controller, nb_iter=nb_iter, n_reset_safecheck=n_reset_safecheck
    )
    return is_grasp_reproducible


def do_touch_variance_computation(
        all_touch_on_obj, all_touch_on_robot, n_steps_continuous_touch, max_observed_n_touch,
        n_steps_discontinuous_touch
):
    #pdb.set_trace()

    entry_points_obj = {}
    for i_entry_point in range(max_observed_n_touch):
        entry_points_obj[i_entry_point] = []
        for pt_on_obj in all_touch_on_obj:
            if len(pt_on_obj) > i_entry_point:
                curr_entry_point = pt_on_obj[i_entry_point]
                entry_points_obj[i_entry_point].append(curr_entry_point)

    entry_points_robot = {}
    for i_entry_point in range(max_observed_n_touch):
        entry_points_robot[i_entry_point] = []
        for pt_on_robot in all_touch_on_robot:
            if len(pt_on_robot) > i_entry_point:
                curr_entry_point = pt_on_robot[i_entry_point]
                entry_points_robot[i_entry_point].append(curr_entry_point)

    all_var_on_obj = []
    for i_entry_point in entry_points_obj:
        pos_on_obj_var_per_coord = np.array(entry_points_obj[i_entry_point]).var(axis=0)
        all_var_on_obj.append(pos_on_obj_var_per_coord)

    all_var_on_robot = []
    for i_entry_point in entry_points_robot:
        pos_on_robot_var_per_coord = np.array(entry_points_obj[i_entry_point]).var(axis=0)
        all_var_on_robot.append(pos_on_robot_var_per_coord)

    summed_variance_on_obj = np.array(all_var_on_obj).sum()
    summed_variance_on_robot = np.array(all_var_on_robot).sum()
    tot_var_per_coord = summed_variance_on_obj + summed_variance_on_robot

    discontinuous_penalty = n_steps_discontinuous_touch
    touch_variance = tot_var_per_coord / (max_observed_n_touch * n_steps_continuous_touch)

    continous_touching_variance = touch_variance + discontinuous_penalty

    #  Fitness = Minimizing criterion => Maximizing -criterion
    fitness_touch_var = (-1) * continous_touching_variance
    return fitness_touch_var

