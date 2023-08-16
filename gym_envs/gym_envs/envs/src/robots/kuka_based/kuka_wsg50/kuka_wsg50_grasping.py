
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

    def __init__(self, object_position=kw_consts.KUKA_DEFAULT_INIT_OBJECT_POSITION, **kwargs):

        table_height = env_consts.TABLE_HEIGHT

        n_control_gripper = self._get_n_dof_gripper()
        end_effector_id = kw_consts.KUKA_END_EFFECTOR_ID
        center_workspace = kw_consts.KUKA_CENTER_WORKSPACE

        ws_radius = ku_consts.KUKA_WS_RADIUS

        disabled_collision_pair = kw_consts.KUKA_DISABLED_COLLISION_PAIRS

        joint_ids = kw_consts.KUKA_JOINT_IDS

        contact_ids = kw_consts.KUKA_CONTACT_IDS

        joint_id_base_left_finger = 8
        joint_id_base_right_finger = 11
        joint_id_base_left_tip_joint = 10
        joint_id_base_right_tip_joint = 13

        jointid_lowerlim_upperlim = [
            (joint_id_base_left_finger, -0.5, -0.05),
            (joint_id_base_right_finger, 0.05, 0.5),
            (joint_id_base_left_tip_joint, -0.3, 0.1),
            (joint_id_base_right_tip_joint, -0.1, 0.3)
        ]

        change_dynamics = {
            **{id: {'lateralFriction': 1,
                    'jointLowerLimit': lowlim,
                    'jointUpperLimit': highlim,
                    'jointLimitForce': 10,
                    'jointDamping': 0.5,
                    'maxJointVelocity': 0.5} for id, lowlim, highlim in jointid_lowerlim_upperlim
            },
            **{i: {'maxJointVelocity': 0.5,
                   'jointLimitForce': 100 if i==1 else 1 if i==6 else 50} for i in range(7)}
        }

        super().__init__(
            robot_class=self._load_model, #load_kuka,
            object_position=object_position,
            table_height=table_height,
            joint_ids=joint_ids,
            contact_ids=contact_ids,
            n_control_gripper=n_control_gripper,
            end_effector_id=end_effector_id,
            center_workspace=center_workspace,
            ws_radius=ws_radius,
            disabled_collision_pair=disabled_collision_pair,
            change_dynamics=change_dynamics,
            is_there_primitive_gene=False,
            **kwargs,
        )

        self.n_dof_arm = len(kw_consts.KUKA_ARM_JOINT_ID_STATUS)


    def step(self, action):

        assert action is not None
        assert len(action) == self.n_actions

        # Update info
        self.info['closed gripper'] = self._is_gripper_closed(action)

        # Convert action to a gym-grasp compatible command
        gripper_command = self._get_gripper_command(action)
        arm_command = action[:-1]
        robot_command = np.hstack([arm_command, gripper_command])

        # Send the command to the robot
        return super().step(robot_command)

    def _load_model(self):
        cwd = Path(gym_envs.__file__).resolve().parent / "envs"
        robot_id = self.p.loadSDF(str(cwd / "3d_models/robots/kuka_iiwa/kuka_gripper_end_effector.sdf"))[0]
        self.p.resetBasePositionAndOrientation(robot_id, [-0.1, -0.5, -0.5], [0., 0., 0., 1.])
        return robot_id

    def _is_gripper_closed(self, action):
        is_closed = action[-1] < 0
        return is_closed

    def _get_gripper_command(self, action):
        # action = [cmd_each_joint, cmd_grip] -> cmd = [cmd_each_joint, -cmd_grip, -cmd_grip, cmd_grip, cmd_grip]
        fingers_cmd = [-action[-1], -action[-1], action[-1], action[-1]]
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




