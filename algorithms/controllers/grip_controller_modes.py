
import numpy as np
from abc import abstractmethod

from .controller_root import ControllerRoot
from . import controller_params as ctrl_params


class GripControllerMode(ControllerRoot):
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

    @abstractmethod
    def get_grip_control_mode(self):
        raise NotImplementedError('Must be overloaded in subclasses.')


class StandardController(GripControllerMode):
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

        self.i_action_grip_close = -1
        assert self.i_action_grip_close in [-1, -2]

    def _get_action_standard_finger_poses(self, i_step):
        if self.grip_time is not None:
            return self.grip_action_open if i_step < self.grip_time else self.grip_action_close
        else:
            return self.grip_action_open

    def get_grip_control_mode(self):
        return ctrl_params.GripControlMode.STANDARD

    @abstractmethod
    def get_action(self, i_step, nrmlized_pos_arm, env):
        raise NotImplementedError('Must be overloaded in subclasses.')


class SynergiesController(GripControllerMode):
    def __init__(self, individual, nb_iter, n_keypoints, n_genes_per_keypoint, n_it_closing_grip,
                 env_name=None, initial=None, a_min=None, a_max=None):

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

        self.i_action_grip_close = -2
        self.i_action_primitive_label = -1
        assert self.i_action_grip_close in [-2, -3]
        assert self.i_action_primitive_label in [-1, -2]

        self.grasp_primitive_label = self._get_grasp_primitive_label(individual)

    def _get_grasp_primitive_label(self, individual):
        #  to refactore
        gpl_supported_values = [0, 1, 2, 3, 4, 5, 6]
        gpl_max_value = 6.999999  #  labels are in {1,2,3,4,5,6}
        # pdb.set_trace()
        gpl_value = individual[self.i_action_primitive_label]
        gpl_value = np.clip(gpl_value, -1, 1)
        assert -1 <= gpl_value <= 1
        gpl_value = (gpl_value + 1) / 2  #  cvt into [0,1]
        gpl_value *= gpl_max_value

        gpl_value = np.clip(np.floor(gpl_value), 0, 6)
        gpl_value = int(gpl_value)
        assert gpl_value in gpl_supported_values

        return gpl_value

    def _get_action_synergies_finger_poses(self, i_step):
        if self.grip_time is not None:
            target_closing_fingers = self.grip_action_open if i_step < self.grip_time else self.grip_action_close
        else:
            target_closing_fingers = self.grip_action_open
        target_pattern_fingers = self.grasp_primitive_label
        target_fingers_pos = [target_closing_fingers, target_pattern_fingers]
        return target_fingers_pos

    def get_grip_control_mode(self):
        return ctrl_params.GripControlMode.WITH_SYNERGIES

    @abstractmethod
    def get_action(self, i_step, nrmlized_pos_arm, env):
        raise NotImplementedError('Must be overloaded in subclasses.')



