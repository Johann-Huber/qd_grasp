
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


"""

    # ---------------------------------------------------------------------------------------- #
    #                              KUKA IIWA ARM + ALLEGRO HAND
    # ---------------------------------------------------------------------------------------- #
    
    # Notes :

    - baselink (29), kuka_joint_a7-end_effector (7), kuka_to_allegro_joint (8), and tips (13, 18, 23, 28) are discarded
    
    - joint_ids : concatenation of arm_joint + hand joints
    - contact_ids : hand ids (9 to 27)
    - n_control_gripper : 16
    - end_effector_id : 7 (literally the end eff id)
    
    - joint_0, joint_4, joint_8 : palm-proximal yaw. Considered fixed here.
    - joint_12 : thumb palm-proximal yaw.
    - joint_1, joint_5, joint_9, joint_13 : palm-proximal pitch.
    - joint_2, joint_6, joint_10, joint_14 : proximal-middle pitch.
    - joint_3, joint_7, joint_11, joint_15 : middle-distal pitch.
    - get_fingers : ndarray associated to each hand joints: 
        [0, -x, -x, -x, 0, -x, -x, -x, 0, -x, -x, -x, -x, -x, -x, -x]

"""


KUKA_ARM_JOINT_ID_STATUS = {
    0:  {'name': 'J0',          'status': 'CONTROLLED',         'is_controllable': True},
    1:  {'name': 'J1',          'status': 'CONTROLLED',         'is_controllable': True},
    2:  {'name': 'J2',          'status': 'CONTROLLED',         'is_controllable': True},
    3:  {'name': 'J3',          'status': 'CONTROLLED',         'is_controllable': True},
    4:  {'name': 'J4',          'status': 'CONTROLLED',         'is_controllable': True},
    5:  {'name': 'J5',          'status': 'CONTROLLED',         'is_controllable': True},
    6:  {'name': 'J6',          'status': 'CONTROLLED',         'is_controllable': True},
}

KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS = {
    9:  {'name': 'joint_0',           'status': 'FIXED',        'part': 'index_finger',   'is_controllable': True},
    10: {'name': 'joint_1',           'status': 'CONTROLLED',   'part': 'index_finger',   'is_controllable': True},
    11: {'name': 'joint_2',           'status': 'CONTROLLED',   'part': 'index_finger',   'is_controllable': True},
    12: {'name': 'joint_3',           'status': 'CONTROLLED',   'part': 'index_finger',   'is_controllable': True},
    13: {'name': 'joint_3_tip',       'status': 'FIXED',        'part': 'index_finger',   'is_controllable': False},

    14: {'name': 'joint_4',           'status': 'FIXED',        'part': 'mid_finger',   'is_controllable': True},
    15: {'name': 'joint_5',           'status': 'CONTROLLED',   'part': 'mid_finger',   'is_controllable': True},
    16: {'name': 'joint_6',           'status': 'CONTROLLED',   'part': 'mid_finger',   'is_controllable': True},
    17: {'name': 'joint_7',           'status': 'CONTROLLED',   'part': 'mid_finger',   'is_controllable': True},
    18: {'name': 'joint_7_tip',       'status': 'FIXED',        'part': 'mid_finger',   'is_controllable': False},

    19: {'name': 'joint_8',           'status': 'FIXED',        'part': 'last_finger',   'is_controllable': True},
    20: {'name': 'joint_9',           'status': 'CONTROLLED',   'part': 'last_finger',   'is_controllable': True},
    21: {'name': 'joint_10',          'status': 'CONTROLLED',   'part': 'last_finger',   'is_controllable': True},
    22: {'name': 'joint_11',          'status': 'CONTROLLED',   'part': 'last_finger',   'is_controllable': True},
    23: {'name': 'joint_11_tip',      'status': 'FIXED',        'part': 'last_finger',   'is_controllable': False},

    24: {'name': 'joint_12',          'status': 'FIXED',        'part': 'thumb',   'is_controllable': True},
    25: {'name': 'joint_13',          'status': 'FIXED',        'part': 'thumb',   'is_controllable': True},
    26: {'name': 'joint_14',          'status': 'CONTROLLED',   'part': 'thumb',   'is_controllable': True},
    27: {'name': 'joint_15',          'status': 'CONTROLLED',   'part': 'thumb',   'is_controllable': True},
}

# Hand closing primitive utils
DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_THUMB = [15, 16, 17, 20, 21, 22]

DISCARD_ALL_JOINT_IDS_BUT_NOT_MID_THUMB = [10, 11, 12, 20, 21, 22]

DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_MID_THUMB = [20, 21, 22]
DISCARD_ALL_JOINT_IDS_BUT_NOT_MID_LAST_THUMB = [10, 11, 12]

DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX = [10, 11]
DISCARD_NO_JOINT_IDS = []

DISCARDED_J_IDS_CLOSING_PRIMITIVES = {
    0: DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_THUMB,
    1: DISCARD_ALL_JOINT_IDS_BUT_NOT_MID_THUMB,
    2: DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_MID_THUMB,
    3: DISCARD_ALL_JOINT_IDS_BUT_NOT_INDEX_MID_THUMB,
    4: DISCARD_ALL_JOINT_IDS_BUT_NOT_MID_LAST_THUMB,
    5: DISCARD_NO_JOINT_IDS,
    6: DISCARD_NO_JOINT_IDS
}


class KukaAllegroGrasping(RobotGrasping):

    def __init__(
            self,
            object_position=[0, 0.1, 0],
            left=False,  # the default is the right hand
            **kwargs
            ):

        cwd = Path(gym_envs.__file__).resolve().parent/"envs"
        urdf = Path(cwd/f"3d_models/robots/generated/kuka_iiwa_allegro_{'left' if left else 'right'}.urdf")
        urdf.parent.mkdir(exist_ok=True, parents=True)
        if not urdf.is_file():  # create the file if doesn't exist
            _process(
                cwd/"3d_models/robots/lbr_iiwa/urdf/lbr_iiwa_14_r820_allegro.xacro",
                dict(output=urdf, just_deps=False, xacro_ns=True, verbosity=1,
                     mappings={'arg_left':'true' if left else 'false'})
            )  # convert xacro to urdf

        def load_kuka():
            kuka_allegro_id = self.p.loadURDF(
                str(urdf), basePosition=[0, -0.5, -0.5], baseOrientation=[0., 0., 0., 1.], useFixedBase=True
            )

            self._print_all_robot_joint_infos(robot_id=kuka_allegro_id)

            return kuka_allegro_id

        controllable_j_ids = self._get_joint_ids()
        n_controlled_joints_gripper = self._get_n_dof_gripper()

        contact_ids = self._get_all_gripper_ids()

        self.i_action_grip_close = -2
        self.i_action_primitive_label = -1

        self.n_dof_arm = 7

        super().__init__(
            robot_class=load_kuka,
            object_position=object_position,
            table_height=0.8,
            joint_ids=controllable_j_ids,
            contact_ids=contact_ids,
            n_control_gripper=n_controlled_joints_gripper,
            end_effector_id=7,
            center_workspace=0,
            ws_radius=1.2,
            disabled_collision_pair=[],
            change_dynamics={**{}, **{i:{'maxJointVelocity': 0.5} for i in range(7)}},
            **kwargs,
            is_there_primitive_gene=True,
        )

    def _get_all_gripper_ids(self):
        return [j_id for j_id in KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS]

    def _is_gripper_closed(self, action):
        is_closed = action[-1] < 0
        return is_closed

    def _get_joint_ids(self):
        # what do we work with, among the controllable joints
        return self.get_arm_controlled_joint_ids() + self.get_gripper_controlled_joint_ids()

    def get_gripper_controlled_joint_ids(self):
        return [
            j_id for j_id in KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS
            if KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
        ]

    def get_fingers_primitives(self, x, primitive_val):
        is_closing = x == -1

        if not is_closing:
            return self.get_fingers(x)
        else:
            discarded_j_ids = DISCARDED_J_IDS_CLOSING_PRIMITIVES[primitive_val]  # add primitive

            # Format when giving only controllable joint to robot_grasping:
            return np.array(
                [(-1 if j_id not in discarded_j_ids else 1) * x
                 for j_id in KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS
                 if KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
                 ]
            )

    def step(self, action=None):
        if action is None:
            return super().step()

        fingers = self.get_fingers_primitives(action[self.i_action_grip_close], action[self.i_action_primitive_label])

        assert(len(action) == self.n_actions)
        self.info['closed gripper'] = action[-1] < 0
        arm = action[:self.i_action_grip_close]
        commands = np.hstack([arm, fingers])

        # apply the commands
        return super().step(commands)

    def _get_arm_controllable_joint_ids(self):
        return [
            j_id for j_id in KUKA_ARM_JOINT_ID_STATUS
            if KUKA_ARM_JOINT_ID_STATUS[j_id]['is_controllable']
        ]

    def _get_n_dof_arm(self):
        return np.array(
            [1 if KUKA_ARM_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED' else 0 for j_id in KUKA_ARM_JOINT_ID_STATUS]
        ).sum()

    def _get_gripper_controllable_joint_ids(self):
        return [
            j_id for j_id in KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS
            if KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['is_controllable']
        ]

    def _get_n_dof_gripper(self):
        return np.array(
            [1 if KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED' else 0
             for j_id in KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS]
        ).sum()

    def get_arm_controlled_joint_ids(self):
        return [
            j_id for j_id in KUKA_ARM_JOINT_ID_STATUS
            if KUKA_ARM_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
        ]

    def _get_controllable_joint_ids(self):
        # what joints the urdf allows us to control
        return self._get_arm_controllable_joint_ids() + self._get_gripper_controllable_joint_ids()

    def _get_rest_poses(self):
        return env_consts.KUKA_ABOVE_OBJECT_INIT_POSITION_JOINTS

    def get_fingers(self, x):
        # Format when giving only controllable joint to robot_grasping:
        return np.array(
            [-x for j_id in KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS
             if KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
             ]
        )

    def _reset_robot(self):

        for j_id in env_consts.KUKA_ALLEGRO_JOINT_IDS:
            self.p.resetJointState(self.robot_id, j_id, targetValue=0.)

        for j_id, j_val in zip(env_consts.KUKA_ALLEGRO_JOINT_IDS, env_consts.KUKA_ABOVE_OBJECT_INIT_POSITION):
            self.p.resetJointState(self.robot_id, j_id, targetValue=j_val)

        #  makes the palm face the table
        self.p.resetJointState(self.robot_id, 6, targetValue=np.pi)

        j_id_thumb_opposition = 24
        HAND_PRESET_VALUES = {
            j_id_thumb_opposition: 1.396,  # opposed thumb
        }

        for j_id in HAND_PRESET_VALUES:
            target_val = HAND_PRESET_VALUES[j_id]
            self.p.resetJointState(self.robot_id, j_id, targetValue=target_val)



