
import pdb

import numpy as np
import gym
import os
import random
from pathlib import Path

from gym_envs.envs.src.robot_grasping import RobotGrasping
from gym_envs.envs.src.xacro import _process

import gym_envs
import gym_envs.envs.src.env_constants as env_consts
import gym_envs.envs.src.robots.kuka_based.kuka_consts as ku_consts
import gym_envs.envs.src.robots.kuka_based.kuka_allegro.kuka_allegro_consts as ka_consts


"""
    # ---------------------------------------------------------------------------------------- #
    #                              KUKA IIWA ARM + ALLEGRO HAND
    # ---------------------------------------------------------------------------------------- #
"""


def generate_urdf_from_xacro(root_3d_models_robots, urdf):
    # create the file if doesn't exist (xacro to urdf conversion)
    xacro2urdf_kwargs = dict(output=urdf, just_deps=False, xacro_ns=True, verbosity=1,
                             mappings={'arg_left': 'false'})
    _process(
        root_3d_models_robots / ka_consts.KUKA_ALLEGRO_RELATIVE_PATH_XACRO,
        xacro2urdf_kwargs
    )


class KukaAllegroGrasping(RobotGrasping):

    def __init__(self, **kwargs):

        controllable_j_ids = self._get_joint_ids()

        kuka_allegro_kwargs = {
            'robot_class': self._load_model,
            'contact_ids': self._get_all_gripper_ids(),
            'n_control_gripper': self._get_n_dof_gripper(),
            'object_position': ka_consts.KUKA_ALLEGRO_DEFAULT_OBJECT_POSE,
            'table_height': ka_consts.KUKA_ALLEGRO_TABLE_HEIGHT,
            'joint_ids': controllable_j_ids,
            'end_effector_id': ka_consts.KUKA_ALLEGRO_END_EFFECTOR_JOINT_ID,
            'center_workspace': ka_consts.KUKA_ALLEGRO_CENTER_WORKSPACE,
            'disabled_collision_pair': ka_consts.KUKA_ALLEGRO_DISABLE_COLLISION_PAIRS,
            'change_dynamics': ka_consts.KUKA_ALLEGRO_CHANGE_DYNAMICS,
            'ws_radius': ku_consts.KUKA_WS_RADIUS,
            'is_there_primitive_gene': True,
        }

        self.i_action_grip_close = -2
        self.i_action_primitive_label = -1
        self.n_dof_arm = len(ka_consts.KUKA_ARM_JOINT_ID_STATUS)

        super().__init__(
            **kuka_allegro_kwargs,
            **kwargs,
        )

    def step(self, action=None):

        assert action is not None
        assert len(action) == self.n_actions

        # Update info
        self.info['closed gripper'] = self._is_gripper_closed(action)

        # Convert action to a gym-grasp compatible command
        gripper_command = self._get_gripper_command_primitives(
            action_gripper_genome_val=action[self.i_action_grip_close],
            primitive_genome_val=action[self.i_action_primitive_label],
        )
        arm_command = action[:self.i_action_grip_close]
        robot_command = np.hstack([arm_command, gripper_command])

        # Send the command to the robot
        return super().step(robot_command)

    def _load_model(self):

        root_3d_models_robots = \
            Path(gym_envs.__file__).resolve().parent / env_consts.GYM_ENVS_RELATIVE_PATH2ROBOTS_MODELS

        urdf = Path(root_3d_models_robots / ka_consts.KUKA_ALLEGRO_RELATIVE_PATH_GENERATED_URDF)

        urdf.parent.mkdir(exist_ok=True, parents=True)
        if not urdf.is_file():
            generate_urdf_from_xacro(root_3d_models_robots=root_3d_models_robots, urdf=urdf)

        kuka_allegro_id = self.p.loadURDF(
            str(urdf),
            basePosition=ka_consts.KUKA_ALLEGRO_BASE_POSITION,
            baseOrientation=ka_consts.KUKA_ALLEGRO_BASE_ORIENTATION,
            useFixedBase=True
        )

        # self._print_all_robot_joint_infos(robot_id=kuka_allegro_id)
        return kuka_allegro_id

    def _get_all_gripper_ids(self):
        return ka_consts.KUKA_ALLEGRO_GRIP_ALL_JOINT_IDS

    def _is_gripper_closed(self, action):
        is_closed = action[-1] < 0
        return is_closed

    def _get_gripper_command_primitives(self, action_gripper_genome_val, primitive_genome_val):
        assert action_gripper_genome_val in ka_consts.VALID_GRIP_COMMAND_VALUES

        is_closing = action_gripper_genome_val == ka_consts.CLOSE_GRIP_COMMAND_VALUE #-1
        if is_closing:
            return self._get_constant_closure_grip_primitives(primitive_label=primitive_genome_val)

        return self._get_open_grip_commands(action_gripper_genome_val)

    def _get_arm_controllable_joint_ids(self):
        return ka_consts.KUKA_ARM_CONTROLLABLE_JOINT_IDS

    def _get_gripper_controllable_joint_ids(self):
        return ka_consts.ALLEGRO_HAND_CONTROLLABLE_JOINT_IDS

    def _get_arm_controlled_joint_ids(self):
        return ka_consts.KUKA_ARM_CONTROLLED_JOINT_IDS

    def _get_gripper_controlled_joint_ids(self):
        return ka_consts.ALLEGRO_HAND_CONTROLLED_JOINT_IDS

    def _get_n_dof_arm(self):
        return len(self._get_arm_controlled_joint_ids())

    def _get_n_dof_gripper(self):
        return len(self._get_gripper_controlled_joint_ids())

    def _get_controllable_joint_ids(self):
        # controllable: what joints the urdf allows us to control
        return self._get_arm_controllable_joint_ids() + self._get_gripper_controllable_joint_ids()

    def _get_joint_ids(self):
        # controlled: what do we work with, among the controllable joints
        return self._get_arm_controlled_joint_ids() + self._get_gripper_controlled_joint_ids()

    def _get_rest_poses(self):
        return ku_consts.KUKA_ABOVE_OBJECT_INIT_POSITION_JOINTS

    def _get_open_grip_commands(self, x):
        # required format when giving only controlled joint to robot_grasping
        return np.array([-x] * len(ka_consts.ALLEGRO_HAND_CONTROLLED_JOINT_IDS))

    def _get_constant_closure_grip_primitives(self, primitive_label):
        discarded_j_ids = ka_consts.DISCARDED_J_IDS_CLOSING_PRIMITIVES[primitive_label]

        # required format when giving only controlled joint to robot_grasping
        return np.array(
            [(-1 if j_id not in discarded_j_ids else 1) * ka_consts.CLOSE_GRIP_COMMAND_VALUE
             for j_id in ka_consts.KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS
             if ka_consts.KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
             ]
        )

    def _set_robot_default_state(self):
        for j_id, j_val in ka_consts.DEFAULT_JOINT_STATES.items():
            self.p.resetJointState(self.robot_id, j_id, targetValue=j_val)

    def _set_robot_manually_fixed_joint_states(self):
        for j_id, j_val in ka_consts.MANUALLY_PRESET_VALUES.items():
            self.p.resetJointState(self.robot_id, j_id, targetValue=j_val)

    def _reset_robot(self):
        self._set_robot_default_state()
        self._set_robot_manually_fixed_joint_states()




