
import numpy as np

from .controller_spaces import JointController, InverseKinematicsController
from .grip_controller_modes import StandardController, SynergiesController


class StandardWayPointsJointController(JointController, StandardController):
    def __init__(self, individual, nb_iter, n_keypoints, n_genes_per_keypoint, n_it_closing_grip, initial=None,
                 env_name=None, a_min=None, a_max=None,  **kwargs):

        #for k in kwargs:
        #    print(f'Unused given key: {k}')

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

    def get_action(self, i_step, nrmlized_pos_arm, env):
        # Get arm joint poses from the interpolated trajectory
        joint_poses_arm = self._get_action_arm_joint_poses(i_step=i_step, nrmlized_pos_arm=nrmlized_pos_arm, env=env)

        # Get the gripper state (constant closure)
        target_pos_fingers = self._get_action_standard_finger_poses(i_step=i_step)

        action = np.append(joint_poses_arm, target_pos_fingers)
        action = self._clip_action(action)
        return action


class StandardWayPointsIKController(InverseKinematicsController, StandardController):
    def __init__(self, individual, nb_iter, n_keypoints, n_genes_per_keypoint, n_it_closing_grip, initial=None,
                 env_name=None, a_min=None, a_max=None, **kwargs):

        #for k in kwargs:
        #    print(f'Unused given key: {k}')

        super().__init__(
            individual=individual,
            nb_iter=nb_iter,
            n_keypoints=n_keypoints,
            n_genes_per_keypoint=n_genes_per_keypoint,
            n_it_closing_grip=n_it_closing_grip,
            initial=initial,
            a_min=a_min,
            a_max=a_max,
            env_name=env_name
        )

    def get_action(self, i_step, nrmlized_pos_arm, env):
        # Get arm joint poses from the interpolated trajectory
        joint_poses_arm = self._get_ik_action_arm_joint_poses(i_step=i_step, nrmlized_pos_arm=nrmlized_pos_arm, env=env)

        # Get the gripper state (constant closure)
        target_pos_fingers = self._get_action_standard_finger_poses(i_step=i_step)

        action = np.append(joint_poses_arm, target_pos_fingers)
        action = self._clip_action(action)
        return action


class SynergiesWayPointsJointController(JointController, SynergiesController):
    def __init__(self, individual, nb_iter, n_keypoints, n_genes_per_keypoint, n_it_closing_grip, initial=None,
                 env_name=None, a_min=None, a_max=None, with_synergies=None, **kwargs):

        assert with_synergies

        #for k in kwargs:
        #    print(f'Unused given key: {k}')

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

    def get_action(self, i_step, nrmlized_pos_arm, env):
        # Get arm joint poses from the interpolated trajectory
        joint_poses_arm = self._get_action_arm_joint_poses(i_step=i_step, nrmlized_pos_arm=nrmlized_pos_arm, env=env)

        # Get the gripper state (constant closure)
        target_pos_fingers = self._get_action_synergies_finger_poses(i_step=i_step)

        action = np.append(joint_poses_arm, target_pos_fingers)
        action = self._clip_action(action)
        return action


class SynergiesWayPointsIKController(InverseKinematicsController, SynergiesController):
    def __init__(self, individual, nb_iter, n_keypoints, n_genes_per_keypoint, n_it_closing_grip,
                 env_name=None, initial=None, a_min=None, a_max=None,
                 with_synergies=None, **kwargs):

        assert with_synergies

        #for k in kwargs:
        #    print(f'Unused given key: {k}')

        super().__init__(
            individual=individual,
            nb_iter=nb_iter,
            n_keypoints=n_keypoints,
            n_genes_per_keypoint=n_genes_per_keypoint,
            n_it_closing_grip=n_it_closing_grip,
            initial=initial,
            a_min=a_min,
            a_max=a_max,
            env_name=env_name
        )

    def get_action(self, i_step, nrmlized_pos_arm, env):
        # Get arm joint poses from the interpolated trajectory
        joint_poses_arm = self._get_ik_action_arm_joint_poses(i_step=i_step, nrmlized_pos_arm=nrmlized_pos_arm, env=env)

        # Get the gripper state (constant closure)
        target_pos_fingers = self._get_action_synergies_finger_poses(i_step=i_step)

        action = np.append(joint_poses_arm, target_pos_fingers)
        action = self._clip_action(action)
        return action

