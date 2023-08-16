
import pdb
import pybullet_data
import numpy as np
import gym
import os
import random
from pathlib import Path
from gym_envs.envs.robot_grasping import RobotGrasping
import gym_envs
from gym_envs.envs.xacro import _process
import gym_envs.envs.env_constants as env_consts
import time

'''
i_joint=0 | (0, b'J0', 0, 7, 6, 1, 0.5, 0.0, -2.96706, 2.96706, 50.0, 10.0, b'lbr_iiwa_link_1', (0.0, 0.0, 1.0), (0.1, 0.0, 0.0875), (0.0, 0.0, 0.0, 1.0), -1)
i_joint=1 | (1, b'J1', 0, 8, 7, 1, 0.5, 0.0, -2.0944, 2.0944, 100.0, 10.0, b'lbr_iiwa_link_2', (0.0, 0.0, 1.0), (0.0, 0.03, 0.08249999999999999), (-9.381873917569987e-07, 0.7071080798588513, 0.707105482510614, 9.381839456086129e-07), 0)
i_joint=2 | (2, b'J2', 0, 9, 8, 1, 0.5, 0.0, -2.96706, 2.96706, 50.0, 10.0, b'lbr_iiwa_link_3', (0.0, 0.0, 1.0), (-0.0003, 0.14549999999862046, -0.04200075117044365), (-9.381873917569987e-07, 0.7071080798588513, 0.707105482510614, -9.381839456086129e-07), 1)
i_joint=3 | (3, b'J3', 0, 10, 9, 1, 0.5, 0.0, -2.0944, 2.0944, 50.0, 10.0, b'lbr_iiwa_link_4', (0.0, 0.0, 1.0), (0.0, -0.03, 0.08550000000000002), (-0.7071080798594737, 0.0, 0.0, 0.7071054825112363), 2)
i_joint=4 | (4, b'J4', 0, 11, 10, 1, 0.5, 0.0, -2.96706, 2.96706, 50.0, 10.0, b'lbr_iiwa_link_5', (0.0, 0.0, 1.0), (0.0, 0.11749999999875532, -0.03400067770634158), (9.381873917569989e-07, 0.7071080798588513, 0.707105482510614, 9.381839456086127e-07), 3)
i_joint=5 | (5, b'J5', 0, 12, 11, 1, 0.5, 0.0, -2.0944, 2.0944, 50.0, 10.0, b'lbr_iiwa_link_6', (0.0, 0.0, 1.0), (-0.0001, -0.021, 0.1394999999999999), (-0.7071080798594737, 0.0, 1.0080490974868132e-23, 0.7071054825112363), 4)
i_joint=6 | (6, b'J6', 0, 13, 12, 1, 0.5, 0.0, -3.05433, 3.05433, 1.0, 10.0, b'lbr_iiwa_link_7', (0.0, 0.0, 1.0), (0.0, 0.0803999999994535, -0.00040029752961337673), (-9.381873917569987e-07, 0.7071080798588513, 0.707105482510614, -9.381839456086129e-07), 5)
i_joint=7 | (7, b'gripper_to_arm', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'base_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.02400000000000004), (0.0, 0.0, 0.0, 1.0), 6)
i_joint=8 | (8, b'base_left_finger_joint', 0, 14, 13, 1, 0.0, 0.0, -0.5, -0.05, 10.0, 1.0, b'left_finger', (0.0, 1.0, 0.0), (0.0, 0.024, 0.04500000000000015), (0.0, 0.024997395914712325, 0.0, 0.9996875162757026), 7)
i_joint=9 | (9, b'left_finger_base_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'left_finger_base', (0.0, 0.0, 0.0), (-0.0009954177603205688, 0.0, 0.04014991667795068), (0.0, 0.12467473338522773, 0.0, 0.992197667229329), 8)
i_joint=10 | (10, b'left_base_tip_joint', 0, 15, 14, 1, 0.0, 0.0, -0.3, 0.1, 10.0, 1.0, b'left_finger_tip', (0.0, 1.0, 0.0), (0.0064011650627963075, 0.0, 0.021752992447456466), (0.0, -0.24740395925452294, 0.0, 0.9689124217106448), 9)
i_joint=11 | (11, b'base_right_finger_joint', 0, 16, 15, 1, 0.0, 0.0, 0.05, 0.5, 10.0, 1.0, b'right_finger', (0.0, 1.0, 0.0), (0.0, 0.024, 0.04500000000000015), (0.0, -0.024997395914712325, 0.0, 0.9996875162757026), 7)
i_joint=12 | (12, b'right_finger_base_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'right_finger_base', (0.0, 0.0, 0.0), (0.0009954177603205688, 0.0, 0.04014991667795068), (0.0, -0.12467473338522773, 0.0, 0.992197667229329), 11)
i_joint=13 | (13, b'right_base_tip_joint', 0, 17, 16, 1, 0.0, 0.0, -0.1, 0.3, 10.0, 1.0, b'right_finger_tip', (0.0, 1.0, 0.0), (-0.0064011650627963075, 0.0, 0.021752992447456466), (0.0, 0.24740395925452294, 0.0, 0.9689124217106448), 12)
i_joint=14 | (14, b'end_effector_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'end_effector_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.19500000000000006), (0.0, 0.0, 0.0, 1.0), 7)
'''

KUKA_ARM_JOINT_ID_STATUS = {
    0:  {'name': 'J0',          'status': 'CONTROLLED',         'is_controllable': True},
    1:  {'name': 'J1',          'status': 'CONTROLLED',         'is_controllable': True},
    2:  {'name': 'J2',          'status': 'CONTROLLED',         'is_controllable': True},
    3:  {'name': 'J3',          'status': 'CONTROLLED',         'is_controllable': True},
    4:  {'name': 'J4',          'status': 'CONTROLLED',         'is_controllable': True},
    5:  {'name': 'J5',          'status': 'CONTROLLED',         'is_controllable': True},
    6:  {'name': 'J6',          'status': 'CONTROLLED',         'is_controllable': True},
}

KUKA_CLAW_GRIP_JOINT_ID_STATUS = {
    7 :   {'name': 'gripper_to_arm',           'status': 'FIXED',        'part': 'gripper',   'is_controllable': False},
    8 :   {'name': 'base_left_finger_joint',   'status': 'CONTROLLED',   'part': 'gripper',   'is_controllable': True},
    9 :   {'name': 'left_finger_base_joint',   'status': 'FIXED',        'part': 'gripper',   'is_controllable': False},
    10:   {'name': 'left_base_tip_joint',      'status': 'CONTROLLED',   'part': 'gripper',   'is_controllable': True},
    11:   {'name': 'base_right_finger_joint',  'status': 'CONTROLLED',   'part': 'gripper',   'is_controllable': True},
    12:   {'name': 'right_finger_base_joint',  'status': 'FIXED',        'part': 'gripper',   'is_controllable': False},
    13:   {'name': 'right_base_tip_joint',     'status': 'CONTROLLED',   'part': 'gripper',   'is_controllable': True},
    14:   {'name': 'end_effector_joint',       'status': 'FIXED',        'part': 'gripper',   'is_controllable': False},
}


class KukaGrasping(RobotGrasping):

    def __init__(self, object_position=env_consts.KUKA_DEFAULT_INIT_OBJECT_POSITION, **kwargs):
        cwd = Path(gym_envs.__file__).resolve().parent / "envs"

        def load_kuka():
            robot_id = self.p.loadSDF(str(cwd/"robots/kuka_iiwa/kuka_gripper_end_effector.sdf"))[0]
            self.p.resetBasePositionAndOrientation(robot_id, [-0.1, -0.5, -0.5], [0., 0., 0., 1.])
            return robot_id

        table_height = env_consts.TABLE_HEIGHT

        n_control_gripper = self._get_n_dof_gripper()
        end_effector_id = env_consts.KUKA_END_EFFECTOR_ID
        center_workspace = env_consts.KUKA_CENTER_WORKSPACE

        ws_radius = env_consts.KUKA_WS_RADIUS

        disabled_collision_pair = env_consts.KUKA_DISABLED_COLLISION_PAIRS

        joint_ids = env_consts.KUKA_JOINT_IDS

        contact_ids = env_consts.KUKA_CONTACT_IDS

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
            robot_class=load_kuka,
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
            j_id for j_id in KUKA_ARM_JOINT_ID_STATUS
            if KUKA_ARM_JOINT_ID_STATUS[j_id]['is_controllable']
        ]

    def _get_gripper_controllable_joint_ids(self):
        return [
            j_id for j_id in KUKA_CLAW_GRIP_JOINT_ID_STATUS
            if KUKA_CLAW_GRIP_JOINT_ID_STATUS[j_id]['is_controllable']
        ]

    def _get_n_dof_gripper(self):
        return np.array(
            [1 if KUKA_CLAW_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED' else 0
             for j_id in KUKA_CLAW_GRIP_JOINT_ID_STATUS]
        ).sum()

    def get_arm_controlled_joint_ids(self):
        return [
            j_id for j_id in KUKA_ARM_JOINT_ID_STATUS
            if KUKA_ARM_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
        ]

    def _get_controllable_joint_ids(self):
        # what joints the urdf allows us to control
        return self._get_arm_controllable_joint_ids() + self._get_gripper_controllable_joint_ids()

    def get_fingers(self, x):
        return np.array([-x, -x, x, x])

    def _get_rest_poses(self):
        return env_consts.KUKA_ABOVE_OBJECT_INIT_POSITION_JOINTS

    def _reset_robot(self):
        for j_id in env_consts.KUKA_JOINT_IDS:
           self.p.resetJointState(self.robot_id, j_id, targetValue=0.)

