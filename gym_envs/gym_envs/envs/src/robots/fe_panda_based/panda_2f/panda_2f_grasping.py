import pdb

import pybullet_data
import numpy as np
import gym
import os
import random
from pathlib import Path

import time

from gym_envs.envs.src.robot_grasping import RobotGrasping
from gym_envs.envs.src.xacro import _process
from gym_envs.envs.src.utils import get_simulation_table_height
import gym_envs
import gym_envs.envs.src.env_constants as env_consts
import gym_envs.envs.src.robots.fe_panda_based.panda_consts as p_consts
import gym_envs.envs.src.robots.fe_panda_based.panda_2f.panda_2f_consts as p2f_consts
import utils.constants as consts

"""
    # ---------------------------------------------------------------------------------------- #
    #                 FRANKA EMIKA PANDA + STANDARD 2-FINGERS PARALLEL GRIPPER
    # ---------------------------------------------------------------------------------------- #
"""


def generate_urdf_from_xacro(root_3d_models_robots, urdf):
    # create the file if doesn't exist (xacro to urdf conversion)
    xacro2urdf_kwargs = dict(output=urdf, just_deps=False, xacro_ns=True, verbosity=1, mappings={})
    _process(
        root_3d_models_robots / p2f_consts.PANDA_2_FINGERS_GRIP_RELATIVE_PATH_XACRO,
        xacro2urdf_kwargs
    )


def init_urdf_franka_emika_panda():
    root_3d_models_robots = \
        Path(gym_envs.__file__).resolve().parent / env_consts.GYM_ENVS_RELATIVE_PATH2ROBOTS_MODELS

    urdf = Path(root_3d_models_robots / p2f_consts.PANDA_2_FINGERS_GRIP_RELATIVE_PATH_GENERATED_URDF)

    urdf.parent.mkdir(exist_ok=True)
    assert urdf.is_file()
    if not urdf.is_file():  # create the file if doesn't exist
        generate_urdf_from_xacro(root_3d_models_robots=root_3d_models_robots, urdf=urdf)

    return urdf


class FrankaEmikaPanda2Fingers(RobotGrasping):

    def __init__(self, **kwargs):
        self.i_action_grip_close = -1
        self.n_dof_arm = self._get_n_dof_arm()

        franka_emika_panda_2_fingers_kwargs = {
            'robot_class': self._load_model,
            'n_control_gripper': self._get_n_dof_gripper(),
            'object_position': p2f_consts.FRANKA_DEFAULT_INIT_OBJECT_POSITION,
            'table_height': env_consts.TABLE_HEIGHT,
            'joint_ids': self._get_joint_ids(),
            'contact_ids': self._get_all_gripper_ids(),
            'end_effector_id': p2f_consts.FRANKA_END_EFFECTOR_JOINT_ID,
            'center_workspace': p2f_consts.PANDA_2_FINGERS_CENTER_WORKSPACE,
            'ws_radius': p2f_consts.PANDA_2_FINGERS_WS_RADIUS,
            'disabled_collision_pair': p2f_consts.PANDA_2_FINGERS_DISABLE_COLLISION_PAIRS,
            'is_there_primitive_gene': False,
            'disabled_obj_robot_contact_ids': self._get_all_arm_ids(),
            'allowed_collision_pair': p2f_consts.PANDA_2_FINGERS_ALLOWED_COLLISION_PAIRS,
        }

        super().__init__(
            **franka_emika_panda_2_fingers_kwargs,
            **kwargs,
        )

    def _load_model(self):
        urdf = init_urdf_franka_emika_panda()

        table_height = env_consts.TABLE_HEIGHT
        floor_table_pos_z = get_simulation_table_height(table_height)
        robot_basis_offset_z = -0.033  # -0.01
        robot_pos_z_on_table = consts.REAL_SCENE_TABLE_TOP + floor_table_pos_z + robot_basis_offset_z

        franka_init_pos = [-0.55, 0.15, robot_pos_z_on_table]

        robot_body_id = self.bullet_client.loadURDF(
            str(urdf),
            basePosition=franka_init_pos,
            baseOrientation=p2f_consts.PANDA_2_FINGERS_BASE_ORIENTATION,
            useFixedBase=True
        )

        #self._print_all_robot_joint_infos(robot_id=robot_body_id)

        return robot_body_id

    def _is_gripper_closed(self, action):
        return action[self.i_action_grip_close] < 0

    def _get_gripper_command(self, action):
        action_grip_genome_val = action[self.i_action_grip_close]
        fingers_cmd = [action_grip_genome_val, action_grip_genome_val]
        return fingers_cmd

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

    def _get_all_arm_ids(self):
        return p2f_consts.FRANKA_EMIKA_PANDA_ARM_ALL_JOINT_IDS

    def _get_all_gripper_ids(self):
        return p2f_consts.PANDA_2_FINGERS_GRIP_ALL_JOINT_IDS

    def _get_n_dof_arm(self):
        return len(self._get_arm_controlled_joint_ids())

    def _get_n_dof_gripper(self):
        return len(self._get_gripper_controlled_joint_ids())

    def _get_arm_controllable_joint_ids(self):
        return p2f_consts.FRANKA_EMIKA_PANDA_ARM_CONTROLLABLE_JOINT_IDS

    def _get_gripper_controllable_joint_ids(self):
        return p2f_consts.PANDA_2_FINGERS_GRIP_CONTROLLABLE_JOINT_IDS

    def _get_arm_controlled_joint_ids(self):
        return p2f_consts.FRANKA_EMIKA_PANDA_ARM_CONTROLLED_JOINT_IDS

    def _get_gripper_controlled_joint_ids(self):
        return p2f_consts.PANDA_2_FINGERS_GRIP_CONTROLLED_JOINT_IDS

    def _get_controllable_joint_ids(self):
        # controllable: what joints the urdf allows us to control
        return self._get_arm_controllable_joint_ids() + self._get_gripper_controllable_joint_ids()

    def _get_joint_ids(self):
        # controlled: what do we work with, among the controllable joints
        return self._get_arm_controlled_joint_ids() + self._get_gripper_controlled_joint_ids()

    def _get_rest_poses(self):
        return p2f_consts.FRANKA_EMIKA_PANDA_INIT_POSITION_JOINTS

    def _set_robot_default_state(self):
        for j_id, pos in p2f_consts.DEFAULT_JOINT_STATES.items():
            self.bullet_client.resetJointState(self.robot_id, j_id, targetValue=pos)

    def _reset_robot(self):
        self._set_robot_default_state()

