import pdb

import numpy as np
from scipy import interpolate
from abc import abstractmethod

from .controller_root import ControllerRoot
from . import controller_params as ctrl_params

import utils.constants as consts


class ControllerSpace(ControllerRoot):
    def __init__(self, individual, nb_iter, n_keypoints, n_genes_per_keypoint, n_it_closing_grip,
                 initial=None, env_name=None, a_min=None, a_max=None):
        super().__init__(
            individual=individual,
            nb_iter=nb_iter,
            n_keypoints=n_keypoints,
            n_genes_per_keypoint=n_genes_per_keypoint,
            n_it_closing_grip=n_it_closing_grip,
            initial=initial,
            a_min=a_min,
            a_max=a_max
        )

    @abstractmethod
    def get_control_space(self):
        raise NotImplementedError('Must be overloaded in subclasses.')

    @abstractmethod
    def get_action(self, i_step, nrmlized_pos_arm, env):
        raise NotImplementedError('Must be overloaded in subclasses.')


class JointController(ControllerSpace):
    def __init__(self, individual, nb_iter, n_keypoints, n_genes_per_keypoint, n_it_closing_grip,
                 initial=None, env_name=None, a_min=None, a_max=None):

        super().__init__(
            individual=individual,
            nb_iter=nb_iter,
            n_keypoints=n_keypoints,
            n_genes_per_keypoint=n_genes_per_keypoint,
            n_it_closing_grip=n_it_closing_grip,
            initial=initial,
            a_min=a_min,
            a_max=a_max,
        )

        self.action_polynome = self._compute_open_loop_trajectory(
            individual=individual,
            nb_iter=nb_iter,
            n_keypoints=n_keypoints,
            initial=initial,
            n_genes_per_keypoint=n_genes_per_keypoint
        )

        self._initialize_grip_time(individual=individual, nb_iter=nb_iter)

    def _extract_joint_actions_from_genome(self, individual, n_keypoints):
        # individual = [arm_poses, t_close, (label_primitive)]
        # extract arm poses : individual[:i_action_grip_close], with i_action_grip_close == t_close
        assert self.i_action_grip_close is not None
        joint_actions_normalized = np.split(np.array(individual), n_keypoints) if consts.T_CLOSE_WHEN_TOUCH_ONLY \
            else np.split(np.array(individual[:self.i_action_grip_close]), n_keypoints)
        return joint_actions_normalized

    def _compute_open_loop_trajectory(self, individual, nb_iter, n_keypoints, initial, n_genes_per_keypoint):

        # Get normalized way points
        actions = self._extract_joint_actions_from_genome(individual=individual, n_keypoints=n_keypoints)

        # Compute key steps to compute interpolation
        n_steps_btw_wp = int(nb_iter / n_keypoints)
        i_steps_interpolate = [int(n_steps_btw_wp / 2 + i * n_steps_btw_wp) for i in range(n_keypoints)]

        # Make the trajectory smoother
        if initial is not None:
            assert len(initial) == n_genes_per_keypoint, \
                f"The length of initial={len(initial)} must be n_genes_per_keypoint={n_genes_per_keypoint}"
            actions.insert(0, initial)
            i_steps_interpolate.insert(0, 0)

        # Compute the open loop trajectory polynome
        action_polynome = interpolate.interp1d(
            i_steps_interpolate, actions, kind='quadratic', axis=0, bounds_error=False, fill_value='extrapolate'
        )
        return action_polynome

    def _get_action_arm_joint_poses(self, i_step, nrmlized_pos_arm, env):

        is_episode_begin = i_step == 0
        if is_episode_begin:
            assert self.last_i == 0

        curr_i = self.last_i if is_episode_begin else self.last_i + 1

        if i_step > self.nb_iter:
            raise RuntimeError(f'Too large i_step ({i_step} > len_episode={self.nb_iter})')

        is_pos_locked = self.lock_end_eff_start_time <= i_step <= self.lock_end_eff_end_time
        # is_pos_locked = True # i_step > 2 #True # debug

        if is_pos_locked:
            # print(f'pos lock: i={i_step}')
            curr_i = self.last_i

        if is_pos_locked and not is_episode_begin:
            target_pos_arm = nrmlized_pos_arm
        else:
            target_pos_arm = self.action_polynome(curr_i)

        self.last_i = curr_i

        return target_pos_arm

    def get_control_space(self):
        return ctrl_params.ControlSpace.JOINT

    @abstractmethod
    def get_action(self, i_step, nrmlized_pos_arm, env):
        raise NotImplementedError('Must be overloaded in subclasses.')


class InverseKinematicsController(ControllerSpace):
    def __init__(self, individual, nb_iter, n_keypoints, n_genes_per_keypoint, n_it_closing_grip, initial=None,
                 env_name=None, a_min=None, a_max=None):

        super().__init__(
            individual=individual,
            nb_iter=nb_iter,
            n_keypoints=n_keypoints,
            n_genes_per_keypoint=n_genes_per_keypoint,
            n_it_closing_grip=n_it_closing_grip,
            initial=initial,
            a_min=a_min,
            a_max=a_max,
        )

        self._env_name = env_name

        self.waypoints_6dof, self.i_step2targeted_wp_list = \
            self._compute_trajectory_ik_way_points(individual=individual)

        self._initialize_grip_time(individual=individual, nb_iter=nb_iter)


        #pdb.set_trace()
        #pass

    def _get_ik_space_lims(self):
        return {
            'min_x_val': consts.CARTESIAN_SCENE_POSE_BOUNDARIES[self._env_name]['min_x_val'],
            'min_y_val': consts.CARTESIAN_SCENE_POSE_BOUNDARIES[self._env_name]['min_y_val'],
            'min_z_val': consts.CARTESIAN_SCENE_POSE_BOUNDARIES[self._env_name]['min_z_val'],

            'max_x_val': consts.CARTESIAN_SCENE_POSE_BOUNDARIES[self._env_name]['max_x_val'],
            'max_y_val': consts.CARTESIAN_SCENE_POSE_BOUNDARIES[self._env_name]['max_y_val'],
            'max_z_val': consts.CARTESIAN_SCENE_POSE_BOUNDARIES[self._env_name]['max_z_val'],
        }

    def _compute_trajectory_ik_way_points(self, individual):
        interp_x = np.linspace(0, self.nb_iter, 4)[1:].astype(int).tolist()

        #  1) convertir les données normalisées en coordonées cartésiennes
        #  [-1; 1] --> [X_MIN, X_MAX]

        #  TANT QU'ON EST EN NORMALISÉ (problème d'exploration difficile fortement contraint)
        approach_keypoint_pose, approach_keypoint_or = individual[:3], individual[3:6]
        prehension_keypoint_pose, prehension_keypoint_or = individual[6:9], individual[9:12]
        verification_keypoint_pose, verification_keypoint_or = individual[12:15], individual[15:18]

        gene_val_min, gene_val_max = -1, 1
        gen_val_range = gene_val_max - gene_val_min

        ik_lims = self._get_ik_space_lims()
        #pdb.set_trace()
        approach_min, approach_max = \
            np.array([ik_lims['min_x_val'], ik_lims['min_y_val'], ik_lims['min_z_val']]), \
            np.array([ik_lims['max_x_val'], ik_lims['max_y_val'], ik_lims['max_z_val']])

        prehension_min, prehension_max = \
            np.array([ik_lims['min_x_val'], ik_lims['min_y_val'], ik_lims['min_z_val']]), \
            np.array([ik_lims['max_x_val'], ik_lims['max_y_val'], ik_lims['max_z_val']])

        verification_min, verification_max = \
            np.array([ik_lims['min_x_val'], ik_lims['min_y_val'], ik_lims['min_z_val']]), \
            np.array([ik_lims['max_x_val'], ik_lims['max_y_val'], ik_lims['max_z_val']])

        pose_approach_range, pose_prehension_range, pose_verification_range = \
            approach_max - approach_min, prehension_max - prehension_min, verification_max - verification_min

        euler_min, euler_max = consts.MIN_EULER_VAL, consts.MAX_EULER_VAL
        euler_range = euler_max - euler_min

        approach_keypoint_pose_xyz = \
            (((approach_keypoint_pose - gene_val_min) * pose_approach_range) / gen_val_range) + approach_min
        approach_keypoint_or_euler = \
            (((approach_keypoint_or - gene_val_min) * euler_range) / gen_val_range) + euler_min

        prehension_keypoint_pose_xyz = \
            (((prehension_keypoint_pose - gene_val_min) * pose_prehension_range) / gen_val_range) + prehension_min
        prehension_keypoint_or_euler = \
            (((prehension_keypoint_or - gene_val_min) * euler_range) / gen_val_range) + euler_min

        verification_keypoint_pose_xyz = \
            (((verification_keypoint_pose - gene_val_min) * pose_verification_range) / gen_val_range) + verification_min
        verification_keypoint_or_euler = \
            (((verification_keypoint_or - gene_val_min) * euler_range) / gen_val_range) + euler_min

        i_step_wp_approach, i_step_wp_prehension, i_step_wp_verification = interp_x[0], interp_x[1], interp_x[2]

        debug_fixed_wp_flg = False
        if debug_fixed_wp_flg:
            approach_keypoint_pose_xyz, approach_keypoint_or_euler, prehension_keypoint_pose_xyz, \
                prehension_keypoint_or_euler, verification_keypoint_pose_xyz, verification_keypoint_or_euler = \
                    self._debug_fixed_way_points()

        #  Note: la commande sera en IK
        waypoints_6dof = {
            0: {
                'xyz': approach_keypoint_pose_xyz,
                'euler': approach_keypoint_or_euler,
                'ref_str': 'approach',
                'i_step': i_step_wp_approach,
            },
            1: {
                'xyz': prehension_keypoint_pose_xyz,
                'euler': prehension_keypoint_or_euler,
                'ref_str': 'prehension',
                'i_step': i_step_wp_prehension,
            },
            2: {
                'xyz': verification_keypoint_pose_xyz,
                'euler': verification_keypoint_or_euler,
                'ref_str': 'verification',
                'i_step': i_step_wp_verification,
            }
        }

        i_step2targeted_wp_list = [0] * i_step_wp_approach + [1] * (i_step_wp_prehension - i_step_wp_approach) + \
                                  [2] * (i_step_wp_verification - i_step_wp_prehension) + \
                                  [2] * (self.nb_iter - i_step_wp_verification)  #  last target is the last wp

        try:
            assert len(i_step2targeted_wp_list) == self.nb_iter
        except:
            pdb.set_trace()

        return waypoints_6dof, i_step2targeted_wp_list

    def _debug_fixed_way_points(self):
        fixed_x = -0.3  #0.0  #-0.3 : baxter  # 0.0 : kuka
        fixed_y = 0.05  #-0.25 : baxter # 0.25 : kuka
        approach_keypoint_pose_xyz = [-0.25, 0., 0.15] #[-0.2, 0.15, 0.15]
        approach_keypoint_or_euler = [0, np.pi/2, 0]
        prehension_keypoint_pose_xyz = [-0.2, -0.05, 0.05]
        prehension_keypoint_or_euler = [0, 2*np.pi/2, 0]
        verification_keypoint_pose_xyz = [0., 0, 0.05] #[0.3, fixed_y, 0.25]
        verification_keypoint_or_euler = [0, 2*np.pi/2, 0]

        return approach_keypoint_pose_xyz, approach_keypoint_or_euler, prehension_keypoint_pose_xyz, \
                prehension_keypoint_or_euler, verification_keypoint_pose_xyz, verification_keypoint_or_euler

    def _get_ik_action_arm_joint_poses(self, i_step, nrmlized_pos_arm, env):

        is_episode_begin = i_step == 0
        if is_episode_begin:
            assert self.last_i == 0

        curr_i = self.last_i if is_episode_begin else self.last_i + 1

        if i_step > self.nb_iter:
            raise RuntimeError(f'Too large i_step ({i_step} > len_episode={self.nb_iter})')

        is_pos_locked = self.lock_end_eff_start_time <= i_step <= self.lock_end_eff_end_time
        #is_pos_locked = True # i_step > 2 #True  # debug

        if is_pos_locked:
            curr_i = self.last_i

        if is_pos_locked and not is_episode_begin:
            target_pos_arm = nrmlized_pos_arm
        else:
            # Select the dof point to target
            target_wp_id = self.i_step2targeted_wp_list[curr_i]
            pos, or_pry = self.waypoints_6dof[target_wp_id]['xyz'], self.waypoints_6dof[target_wp_id]['euler']

            # Apply inverse kinematics
            target_pos_arm = env.get_joints_poses_from_ik(pos, or_pry, normalized=True)

        self.last_i = curr_i

        return target_pos_arm

    def get_control_space(self):
        return ctrl_params.ControlSpace.CARTESIAN

    @abstractmethod
    def get_action(self, i_step, nrmlized_pos_arm, env):
        raise NotImplementedError('Must be overloaded in subclasses.')


