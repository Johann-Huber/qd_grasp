
import pdb

import pybullet_data
import time
import numpy as np
import gym
import os
import random
from pathlib import Path

from gym_envs.envs.src.robot_grasping import RobotGrasping
import gym_envs
from gym_envs.envs.src.xacro import _process
import gym_envs.envs.src.env_constants as env_consts
import gym_envs.envs.src.robots.kuka_based.kuka_consts as ku_consts
import gym_envs.envs.src.robots.kuka_based.kuka_wsg50.kuka_wsg50_consts as kw_consts

"""

    # ---------------------------------------------------------------------------------------- #
    #                              KUKA IIWA ARM + WR50 2-fingers gripper
    # ---------------------------------------------------------------------------------------- #

"""


class KukaWsg50Grasping(RobotGrasping):

    def __init__(self, **kwargs):

        self.i_action_grip_close = -1
        self.n_dof_arm = len(kw_consts.KUKA_ARM_JOINT_ID_STATUS)

        kuka_wsg50_kwargs = {
            'robot_class': self._load_model,
            'n_control_gripper': self._get_n_dof_gripper(),
            'object_position': kw_consts.KUKA_DEFAULT_INIT_OBJECT_POSITION,
            'table_height': env_consts.TABLE_HEIGHT,
            'joint_ids': kw_consts.KUKA_JOINT_IDS,
            'contact_ids': kw_consts.KUKA_CONTACT_IDS,
            'end_effector_id':  kw_consts.KUKA_END_EFFECTOR_ID,
            'center_workspace': kw_consts.KUKA_CENTER_WORKSPACE,
            'ws_radius': ku_consts.KUKA_WS_RADIUS,
            'disabled_collision_pair': kw_consts.KUKA_DISABLED_COLLISION_PAIRS,
            'change_dynamics': kw_consts.KUKA_WSG50_CHANGE_DYNAMICS,
            'is_there_primitive_gene': False,
        }

        super().__init__(
            **kuka_wsg50_kwargs,
            **kwargs,
        )



    def step(self, action):

        assert action is not None
        assert len(action) == self.n_actions

        # Update info
        self.info['closed gripper'] = self._is_gripper_closed(action)

        # Convert action to a gym-grasp compatible command
        gripper_command = self._get_gripper_command(action)
        arm_command = action[:self.i_action_grip_close]
        robot_command = np.hstack([arm_command, gripper_command])

        # Send the command to the robot
        return super().step(robot_command)

    def _load_model(self):
        cwd = Path(gym_envs.__file__).resolve().parent / "envs"
        robot_id = self.p.loadSDF(str(cwd / "3d_models/robots/kuka_iiwa/kuka_gripper_end_effector.sdf"))[0]
        self.p.resetBasePositionAndOrientation(
            robot_id,
            kw_consts.KUKA_WSG50_BASE_POSITION,
            kw_consts.KUKA_WSG50_BASE_ORIENTATION
        )
        return robot_id


    def _is_gripper_closed(self, action):
        is_closed = action[self.i_action_grip_close] < 0
        return is_closed

    def _get_gripper_command(self, action):
        # action = [cmd_each_joint, cmd_grip] -> cmd = [cmd_each_joint, -cmd_grip, -cmd_grip, cmd_grip, cmd_grip]
        action_grip_genome_val = action[self.i_action_grip_close]
        fingers_cmd = [-action_grip_genome_val, -action_grip_genome_val, action_grip_genome_val, action_grip_genome_val]
        return fingers_cmd

    def _get_arm_controllable_joint_ids(self):
        return kw_consts.KUKA_ARM_CONTROLLABLE_JOINT_IDS

    def _get_gripper_controllable_joint_ids(self):
        return kw_consts.WSG50_GRIP_CONTROLLABLE_JOINT_IDS

    def _get_arm_controlled_joint_ids(self):
        return kw_consts.KUKA_ARM_CONTROLLED_JOINT_IDS

    def _get_gripper_controlled_joint_ids(self):
        return kw_consts.WSG50_GRIP_CONTROLLED_JOINT_IDS

    def _get_controllable_joint_ids(self):
        # what joints the urdf allows us to control
        return self._get_arm_controllable_joint_ids() + self._get_gripper_controllable_joint_ids()

    def _get_n_dof_gripper(self):
        return len(self._get_gripper_controlled_joint_ids())

    def _get_rest_poses(self):
        return ku_consts.KUKA_ABOVE_OBJECT_INIT_POSITION_JOINTS

    def _set_robot_default_state(self):
        for j_id, j_val in kw_consts.DEFAULT_JOINT_STATES.items():
            self.p.resetJointState(self.robot_id, j_id, targetValue=j_val)

    def _reset_robot(self):
        self._set_robot_default_state()




