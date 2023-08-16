
from abc import ABC, abstractmethod
import utils.constants as consts
import numpy as np

from . import controller_params as ctrl_params


class ControllerRoot(ABC):
    def __init__(self, nb_iter, n_it_closing_grip, a_min=None, a_max=None, env_name=None,
                 **kwargs):

        self.nb_iter = nb_iter  # episode length

        self.a_min = a_min  # None : risky for real robot but easier to prototype
        self.a_max = a_max  # None

        self.n_it_closing = None  # Current
        self.n_it_closing_grip = n_it_closing_grip  # Number of step for this gripper to close

        self.last_i = 0
        self.grip_action_open = 1
        self.grip_action_close = -1

        self.i_action_grip_close = None

        self.grip_time = None
        self.lock_end_eff_start_time = None
        self.lock_end_eff_end_time = None


    def _clip_action(self, action):
        return np.clip(action, min=self.a_min, max=self.a_max) if (self.a_min is not None or self.a_max is not None) \
            else action

    def reset_rolling_attributes(self):
        self.last_i = 0

    def update_grip_time(self, grip_time):
        self.grip_time = grip_time
        self.lock_end_eff_start_time = self.grip_time - ctrl_params.NB_ITER_LOCK_BEFORE_START_GRASP
        self.lock_end_eff_end_time = self.grip_time + self.n_it_closing_grip

    @abstractmethod
    def get_action(self, i_step, nrmlized_pos_arm, env):
        raise NotImplementedError('Must be overloaded in subclasses.')

