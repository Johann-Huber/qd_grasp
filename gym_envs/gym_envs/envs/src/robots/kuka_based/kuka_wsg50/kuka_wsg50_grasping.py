
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

        ws_radius = kw_consts.KUKA_WS_RADIUS

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

        self.n_dof_arm = 7

    def _load_model(self):
        cwd = Path(gym_envs.__file__).resolve().parent / "envs"
        robot_id = self.p.loadSDF(str(cwd / "3d_models/robots/kuka_iiwa/kuka_gripper_end_effector.sdf"))[0]
        self.p.resetBasePositionAndOrientation(robot_id, [-0.1, -0.5, -0.5], [0., 0., 0., 1.])
        return robot_id

    def _is_gripper_closed(self, action):
        # action = [joint_pos_1, ..., joint_pos_n, grip_pos_1, grip_pos_2, grip_pos_3, grip_pos_4]
        # open grip : grip_pos = [-1, -1, 1, 1]
        # closed grip : grip_pos = [1, 1, -1, -1]
        #print('action=', action)
        is_closed = action[-2] < 0
        assert action[-4] == action[-3] == -action[-2] == -action[-1]
        return is_closed

    def step(self, action):
        # action = [cmd_each_joint, cmd_grip] -> cmd = [cmd_each_joint, -cmd_grip, -cmd_grip, cmd_grip, cmd_grip]
        fingers_cmd = -action[-1], -action[-1], action[-1], action[-1]

        # we want one action per joint (gripper is composed by 4 joints but considered as one)
        assert len(action) == self.n_actions

        griper_id = -1

        self.info['closed gripper'] = action[griper_id] < 0

        commands = np.hstack([action[:-1], *fingers_cmd])

        # apply the commands
        return super().step(commands)

    def _get_arm_controllable_joint_ids(self):
        return [
            j_id for j_id in kw_consts.KUKA_ARM_JOINT_ID_STATUS
            if kw_consts.KUKA_ARM_JOINT_ID_STATUS[j_id]['is_controllable']
        ]

    def _get_gripper_controllable_joint_ids(self):
        return [
            j_id for j_id in kw_consts.KUKA_CLAW_GRIP_JOINT_ID_STATUS
            if kw_consts.KUKA_CLAW_GRIP_JOINT_ID_STATUS[j_id]['is_controllable']
        ]

    def _get_n_dof_gripper(self):
        return np.array(
            [1 if kw_consts.KUKA_CLAW_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED' else 0
             for j_id in kw_consts.KUKA_CLAW_GRIP_JOINT_ID_STATUS]
        ).sum()

    def get_arm_controlled_joint_ids(self):
        return [
            j_id for j_id in kw_consts.KUKA_ARM_JOINT_ID_STATUS
            if kw_consts.KUKA_ARM_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
        ]

    def _get_controllable_joint_ids(self):
        # what joints the urdf allows us to control
        return self._get_arm_controllable_joint_ids() + self._get_gripper_controllable_joint_ids()

    def get_fingers(self, x):
        return np.array([-x, -x, x, x])

    def _get_rest_poses(self):
        return ku_consts.KUKA_ABOVE_OBJECT_INIT_POSITION_JOINTS

    def _reset_robot(self):
        for j_id in kw_consts.KUKA_JOINT_IDS:
           self.p.resetJointState(self.robot_id, j_id, targetValue=0.)


