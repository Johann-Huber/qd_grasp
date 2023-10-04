
import numpy as np
import utils.constants as consts

from algorithms.evaluate import do_traj_redeploiment_safecheck


def compute_object_state_domain_randomization_fit(env, nb_iter, ind, controller_class, controller_info, add_iter,
                                                  force_state_load=False):
    n_noisy_target = consts.N_NOISY_TARGETS

    init_pos_var = consts.OBJ_STATE_DR_INIT_POS_VARIANCE_IN_M
    init_euler_orient_var = consts.OBJ_STATE_DR_INIT_ORIENT_EULER_VARIANCE_IN_RAD

    all_init_rand_pos = np.random.normal(loc=0.0, scale=init_pos_var, size=(n_noisy_target, 2))
    all_init_noise_norm = [np.sqrt(pos[0] ** 2 + pos[1] ** 2) for pos in all_init_rand_pos]

    all_init_rand_orient_euler = np.random.normal(loc=0.0, scale=init_euler_orient_var, size=(n_noisy_target, 3))

    all_is_there_grasp = [
        int(
            do_traj_redeploiment_safecheck(
                env=env,
                nb_iter=nb_iter,
                n_reset_safecheck=1,
                obj_init_state_offset={
                    'init_pos_offset': init_rand_pos,
                    'init_orient_offset_euler': init_rand_orient_euler,
                },
                force_state_load=force_state_load,
                ind=ind,
                controller_class=controller_class,
                controller_info=controller_info,
                add_iter=add_iter
            )
        ) for i_noisy_target, (init_rand_pos, init_rand_orient_euler) in enumerate(
            zip(all_init_rand_pos, all_init_rand_orient_euler)
        )
    ]

    robustness = np.sum([is_scs * noise_norm for is_scs, noise_norm in zip(all_is_there_grasp, all_init_noise_norm)])
    return robustness


def compute_joint_states_domain_randomization_fit(env, add_iter, nb_iter, ind, controller_class, controller_info,
                                          force_state_load=False):
    n_noisy_trajs = consts.N_NOISY_JOINTS_DEPLOYEMENTS

    all_is_there_grasp = [
        int(
            do_traj_redeploiment_safecheck(
                env=env,
                nb_iter=nb_iter,
                n_reset_safecheck=1,
                do_noise_joints_pos=True,
                force_state_load=force_state_load,
                add_iter=add_iter,
                ind=ind, controller_class=controller_class, controller_info=controller_info,
            )
        ) for _ in range(n_noisy_trajs)
    ]

    robustness_noise_joint = np.array(all_is_there_grasp).mean()
    return robustness_noise_joint


def compute_friction_domain_randomization_fit(env, add_iter, nb_iter, ind, controller_class, controller_info,
                                          force_state_load=False):
    n_noisy_trajs = consts.N_NOISY_DYNAMICS_DEPLOYEMENTS

    all_is_there_grasp = [
        int(
            do_traj_redeploiment_safecheck(
                env=env,
                nb_iter=nb_iter,
                n_reset_safecheck=1,
                do_noise_dynamics=True,
                force_state_load=force_state_load,
                ind=ind, controller_class=controller_class, controller_info=controller_info,
                add_iter=add_iter
            )
        ) for _ in range(n_noisy_trajs)
    ]

    robustness_noise_dynamics = np.array(all_is_there_grasp).mean()
    return robustness_noise_dynamics


def compute_mixture_domain_randomization_fit(env, nb_iter, ind, controller_class, controller_info, add_iter,
                              force_state_load=False):
    n_noisy_trajs = consts.N_NOISY_MIXTURE

    init_pos_var = consts.OBJ_STATE_DR_INIT_POS_VARIANCE_IN_M
    init_euler_orient_var = consts.OBJ_STATE_DR_INIT_ORIENT_EULER_VARIANCE_IN_RAD

    all_init_rand_pos = np.random.normal(loc=0.0, scale=init_pos_var, size=(n_noisy_trajs, 2))
    all_init_rand_orient_euler = np.random.normal(loc=0.0, scale=init_euler_orient_var, size=(n_noisy_trajs, 3))

    all_is_there_grasp = [
        int(
            do_traj_redeploiment_safecheck(
                env=env,
                nb_iter=nb_iter,
                n_reset_safecheck=1,
                do_noise_joints_pos=True,
                do_noise_dynamics=True,
                obj_init_state_offset={
                    'init_pos_offset': init_rand_pos,
                    'init_orient_offset_euler': init_rand_orient_euler,
                },
                force_state_load=force_state_load,
                ind=ind,
                controller_class=controller_class,
                controller_info=controller_info,
                add_iter=add_iter,
            )
        ) for i_noisy_target, (init_rand_pos, init_rand_orient_euler) in enumerate(
            zip(all_init_rand_pos, all_init_rand_orient_euler)
        )
    ]

    robustness = np.mean(all_is_there_grasp)
    return robustness


def get_all_quality_criteria(
        interaction_measures, is_there_grasp, individual, env, controller, nb_iter, **kwargs
):
    im = interaction_measures
    controller_class = kwargs['controller_class']
    controller_info = kwargs['controller_info']
    add_iter = kwargs['add_iter']
    is_already_touched = im.is_already_touched
    no_contact_table = kwargs['no_contact_table']
    robot_has_touched_table = im.robot_has_touched_table

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

    print('[OSDR] Quality criteria computation: Domain Randomization on perception (object state)')
    robustness = compute_object_state_domain_randomization_fit(
        env=env, nb_iter=nb_iter, force_state_load=True, ind=individual,
        controller_class=controller_class, controller_info=controller_info, add_iter=add_iter
    )

    print('[JSDR] Quality criteria computation: Domain Randomization on control (joint states)')
    robustness_noise_joint = compute_joint_states_domain_randomization_fit(
        env=env, add_iter=add_iter, nb_iter=nb_iter, force_state_load=True, ind=individual,
        controller_class=controller_class, controller_info=controller_info
    )

    print('[FDR] Quality criteria computation: Domain Randomization on dynamics (friction coefficients)')
    robustness_dynamics = compute_friction_domain_randomization_fit(
        env=env, add_iter=add_iter, nb_iter=nb_iter, force_state_load=True, ind=individual,
        controller_class=controller_class, controller_info=controller_info
    )

    print('[MDR] Quality criteria computation: Domain Randomization on perception, control and dynamics')
    mixture_dr = compute_mixture_domain_randomization_fit(
        env=env, nb_iter=nb_iter, ind=individual, controller_class=controller_class,
        controller_info=controller_info, add_iter=add_iter, force_state_load=True
    )

    # Build quality criteria output
    quality_criteria = {
        'is_valid': is_valid,
        'is_success': is_success,
        'is_obj_touched': is_already_touched,
        'robustness': robustness,
        'robustness_noise_joint': robustness_noise_joint,
        'robustness_dynamics': robustness_dynamics,
        'mixture_dr': mixture_dr,
    }

    return quality_criteria


