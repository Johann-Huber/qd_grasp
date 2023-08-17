

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

from dataclasses import dataclass, field


@dataclass
class InteractionMeasures:
    is_already_grasped: bool = False
    robot_has_touched_table: bool = False
    i_start_closing: int = None
    pos_touch_time: bool = None
    is_already_touched: bool = False
    n_steps_before_grasp: int = consts.INF_FLOAT_CONST
    t_touch_t_close_diff: float = consts.INF_FLOAT_CONST
    curr_contact_object_table: list = field(default_factory=list)


@dataclass
class TouchVarFitMeasures:
    all_touch_on_obj: list = field(default_factory=list)
    all_touch_on_robot: list = field(default_factory=list)
    is_discontinuous_touch: bool = False
    n_steps_continuous_touch: int = 0
    n_steps_discontinuous_touch: int = 0
    max_observed_n_touch: int = 0


@dataclass
class EnergyFitMeasure:
    energy_consumption_cumulated: int = 0


def evaluate_grasp_ind(individual, env, eval_kwargs):

    episode_length = eval_kwargs['nb_iter']
    env.reset(load='state' if consts.RESET_MODE else 'all')

    controller = init_controller(
        individual=individual,
        controller_class=eval_kwargs['controller_class'],
        controller_info=eval_kwargs['controller_info'],
        env=env
    )

    im = InteractionMeasures()
    tvfm = TouchVarFitMeasures()
    efm = EnergyFitMeasure()

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
                im.i_start_closing = i_step  # the object should be grasped after having closed the gripper

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
            return get_autocollide_outputs(
                reward_cumulated=reward_cumulated,
                is_already_touched=im.is_already_touched,
                **eval_kwargs
            )

        if im.is_already_touched and not tvfm.is_discontinuous_touch:
            touch_points_on_obj = info['touch_points_obj']
            touch_points_on_robot = info['touch_points_robot']
            is_discontinuous_touch_detected = \
                len(touch_points_on_obj) < tvfm.max_observed_n_touch or \
                len(touch_points_on_robot) < tvfm.max_observed_n_touch
            if is_discontinuous_touch_detected:
                tvfm.is_discontinuous_touch = True
            else:
                tvfm.n_steps_continuous_touch += 1
                tvfm.max_observed_n_touch = \
                    max(tvfm.max_observed_n_touch, len(touch_points_on_obj))
                tvfm.all_touch_on_obj.append(touch_points_on_obj)
                tvfm.all_touch_on_robot.append(touch_points_on_robot)

        if tvfm.is_discontinuous_touch:
            tvfm.n_steps_discontinuous_touch += 1

        reward_cumulated += reward
        efm.energy_consumption_cumulated += np.abs(info['applied joint motor torques']).sum()

        is_robot_touching_table = len(info['contact robot table']) > 0
        im.robot_has_touched_table = im.robot_has_touched_table or is_robot_touching_table

        grasped_while_closing = im.t_touch_t_close_diff < eval_kwargs['n_it_closing_grip'] * consts.GRASP_WHILE_CLOSE_TOLERANCE
        obj_not_touching_table = len(im.curr_contact_object_table) > 0
        is_there_grasp = reward and grasped_while_closing and obj_not_touching_table
        is_first_grasp = is_there_grasp and im.n_steps_before_grasp == consts.INF_FLOAT_CONST
        if is_first_grasp:
            im.n_steps_before_grasp = i_step

        if env.display:
            time.sleep(consts.TIME_SLEEP_SMOOTH_DISPLAY_IN_SEC)

        nrmlized_pos_arm_prev = nrmlized_pos_arm

        i_step += 1

    # use last info to compute behavior and fitness
    grasped_while_closing = im.t_touch_t_close_diff < eval_kwargs['n_it_closing_grip'] * consts.GRASP_WHILE_CLOSE_TOLERANCE
    obj_not_touching_table = len(im.curr_contact_object_table) > 0
    is_there_grasp = reward and grasped_while_closing and obj_not_touching_table

    if is_there_grasp:
        is_there_grasp = do_safechecks_traj_success(
            env=env, controller=controller,
            **eval_kwargs
        )

    return get_evaluate_grasp_ind_outputs(
        interaction_measures=im,
        touch_var_fit_measure=tvfm,
        energy_fit_measure=efm,
        is_there_grasp=is_there_grasp,
        reward_cumulated=reward_cumulated,
        **eval_kwargs)


def exception_handler_evaluate_grasp_ind(individual, eval_kwargs):

    evo_process = eval_kwargs['evo_process']
    prehension_criteria = eval_kwargs['prehension_criteria']
    algo_variant = eval_kwargs['algo_variant']
    bd_len = eval_kwargs['bd_len']

    print(f'DANGER : pathological case raised in eval. ind={individual}')

    # Build behavior
    dummy_behavior_failure = bd_len * [None]

    # Build fitness
    dummy_fitness_failure = build_deap_compatible_fitness(
        scalar_fit=None,
        evo_process=evo_process,
        algo_variant=algo_variant
    )

    # Build info_out
    dummy_info_out_failure = {
        'is_success': False,
        'is_valid': None,
        'energy': 0,
        'n_steps_before_grasp': consts.INF_FLOAT_CONST,
        'reward_cumulated': 0,
        'normalized_multi_fit': 0.,
        'touch_var': -consts.INF_FLOAT_CONST if 'touch_var' in prehension_criteria else None,
    }
    return dummy_behavior_failure, dummy_fitness_failure, dummy_info_out_failure


def get_evaluate_grasp_ind_outputs(
        interaction_measures, robot,
        touch_var_fit_measure, energy_fit_measure, is_there_grasp, reward_cumulated,
        **kwargs # todo à transformer en mesure
):
    im = interaction_measures
    is_already_touched = im.is_already_touched
    evo_process = kwargs['evo_process']
    no_contact_table = kwargs['no_contact_table']
    prehension_criteria = kwargs['prehension_criteria']
    algo_variant = kwargs['algo_variant']
    robot_has_touched_table = im.robot_has_touched_table,
    n_steps_before_grasp = im.n_steps_before_grasp,

    # Sanity check
    is_success = False
    is_valid = True
    if is_there_grasp:
        if not im.is_already_touched:
            is_success = False
            is_valid = False
            assert im.pos_touch_time is None
        else:
            if no_contact_table and robot_has_touched_table:
                is_success = False
                is_valid = False
            else:
                is_success = True
                is_valid = True

    # Fitness postprocessing
    touch_var_fit = get_touch_var_fit(
        is_success=is_success,
        touch_var_fit_measure=touch_var_fit_measure,
        prehension_criteria=prehension_criteria
    )

    energy_fit = get_energy_fit(energy_fit_measure)

    normalized_multi_fit = get_normalized_multi_fitness(
        energy_fit=energy_fit, touch_var_fit=touch_var_fit, robot_name=robot
    )

    # Build behavior
    behavior = build_behavior_vector(
        interaction_measures=im,
        **kwargs
    )

    # Build fitness
    binary_success_fitness = float(is_success)
    fitness = build_deap_compatible_fitness(
        scalar_fit=binary_success_fitness,
        evo_process=evo_process,
        algo_variant=algo_variant
    )

    # Build info_out
    info_out = {
        'is_valid': is_valid,
        'is_success': is_success,
        'auto_collided': False,
        'energy': energy_fit,
        'is_obj_touched': is_already_touched,
        'normalized_multi_fit': normalized_multi_fit,
        'touch_var': touch_var_fit,
        'reward_cumulated': reward_cumulated,
        'n_steps_before_grasp': n_steps_before_grasp
    }

    return behavior, fitness, info_out


def get_touch_var_fit(is_success, touch_var_fit_measure, prehension_criteria):
    if is_success:
        touch_var_fit = do_touch_variance_computation(
            touch_var_fit_measure=touch_var_fit_measure,
        ) if 'touch_var' in prehension_criteria else None
    else:
        touch_var_fit = -consts.INF_FLOAT_CONST if 'touch_var' in prehension_criteria else None
    return touch_var_fit


def get_energy_fit(energy_fit_measure):
    # minimize energy => maximise energy_fit = (-1) * energy_consumption_cumulated
    return (-1) * energy_fit_measure.energy_consumption_cumulated


def get_autocollide_outputs(evo_process, bd_len, algo_variant, robot, is_already_touched, reward_cumulated, verbose=True, **kwargs):

    if verbose:
        print('AUTOCOLLIDE !' + 20 * '=')

    worst_energy = env_consts.ENERGY_FIT_NORM_BOUNDS_PER_ROB[robot]['min']

    worst_touch_var = -consts.INF_FLOAT_CONST
    worst_normalized_multi_fit = 0.
    worst_n_steps_before_grasp = consts.INF_FLOAT_CONST

    # Build behavior descriptor
    behavior = build_invalid_behavior_vector(evo_process=evo_process, bd_len=bd_len)

    # Build fitness
    worst_fitness = -consts.INF_FLOAT_CONST
    fitness = build_deap_compatible_fitness(
        scalar_fit=worst_fitness,
        evo_process=evo_process,
        algo_variant=algo_variant
    )

    # Build info_out
    info_out = {
        'is_valid': False,
        'is_success': False,
        'auto_collided': True,
        'energy': worst_energy,
        'is_obj_touched': is_already_touched,
        'normalized_multi_fit': worst_normalized_multi_fit,
        'touch_var': worst_touch_var,
        'reward_cumulated': reward_cumulated,
        'n_steps_before_grasp': worst_n_steps_before_grasp
    }

    return behavior, fitness, info_out


def build_deap_compatible_fitness(scalar_fit, evo_process, algo_variant):

    if evo_process in consts.EVO_PROCESS_WITH_LOCAL_COMPETITION:
        # NSLC: fit = (nov, local_quality) -> makes NSGA-II selection easily applicable
        return (scalar_fit, scalar_fit)
    elif algo_variant in consts.ARCHIVE_BASED_NOVELTY_FITNESS_SELECTION_ALGO_VARIANTS:
        # ME-nov-fit : fit = (nov, fit) -> makes NSGA-II selection easily applicable
        return (scalar_fit, scalar_fit)
    else:
        # DEAP is expected tuple, even with a single value
        return (scalar_fit,)


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


def build_behavior_vector(bd_flg, evo_process, interaction_measures, bd_len, **kwargs):
    im = interaction_measures
    behavior = np.array([None] * bd_len, dtype=object)
    pos_touch_time = im.pos_touch_time

    if bd_flg == 'pos_touch':
        behavior[:] = pos_touch_time
        behavior = fillna_behavior(behavior=behavior, evo_process=evo_process)
        return behavior.tolist()
    else:
        raise RuntimeError(f'Unknown bd_flg: {bd_flg}')



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


def do_safechecks_traj_success(robot, env, add_iter, controller, nb_iter, n_reset_safecheck, **kwargs):
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


def do_touch_variance_computation(touch_var_fit_measure):

    tvfm = touch_var_fit_measure

    all_touch_on_obj = tvfm.all_touch_on_obj
    all_touch_on_robot = tvfm.all_touch_on_robot
    n_steps_continuous_touch = tvfm.n_steps_continuous_touch
    max_observed_n_touch = tvfm.max_observed_n_touch
    n_steps_discontinuous_touch = tvfm.n_steps_discontinuous_touch

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

