
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


def generate_urdf_from_xacro(root_3d_models_robots, urdf):
    # create the file if doesn't exist
    xacro2urdf_kwargs = dict(output=urdf, just_deps=False, xacro_ns=True, verbosity=1,
                             mappings={'arg_left': 'false'})
    _process(
        root_3d_models_robots / ka_consts.KUKA_ALLEGRO_RELATIVE_PATH_XACRO,
        xacro2urdf_kwargs
    )  # convert xacro to urdf


class KukaAllegroGrasping(RobotGrasping):

    def __init__(
            self,
            object_position=[0, 0.1, 0],
            **kwargs
            ):

        controllable_j_ids = self._get_joint_ids()
        n_controlled_joints_gripper = self._get_n_dof_gripper()

        contact_ids = self._get_all_gripper_ids()

        self.i_action_grip_close = -2
        self.i_action_primitive_label = -1

        self.n_dof_arm = 7

        super().__init__(
            robot_class=self._load_model,
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
        return [j_id for j_id in ka_consts.KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS]

    def _is_gripper_closed(self, action):
        is_closed = action[-1] < 0
        return is_closed

    def _get_joint_ids(self):
        # what do we work with, among the controllable joints
        return self.get_arm_controlled_joint_ids() + self.get_gripper_controlled_joint_ids()

    def get_gripper_controlled_joint_ids(self):
        return [
            j_id for j_id in ka_consts.KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS
            if ka_consts.KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
        ]

    def get_fingers_primitives(self, x, primitive_val):
        is_closing = x == -1

        if not is_closing:
            return self.get_fingers(x)
        else:
            discarded_j_ids = ka_consts.DISCARDED_J_IDS_CLOSING_PRIMITIVES[primitive_val]  # add primitive

            # Format when giving only controllable joint to robot_grasping:
            return np.array(
                [(-1 if j_id not in discarded_j_ids else 1) * x
                 for j_id in ka_consts.KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS
                 if ka_consts.KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
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
            j_id for j_id in ka_consts.KUKA_ARM_JOINT_ID_STATUS
            if ka_consts.KUKA_ARM_JOINT_ID_STATUS[j_id]['is_controllable']
        ]

    def _get_n_dof_arm(self):
        return np.array(
            [1 if ka_consts.KUKA_ARM_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED' else 0 for j_id in ka_consts.KUKA_ARM_JOINT_ID_STATUS]
        ).sum()

    def _get_gripper_controllable_joint_ids(self):
        return [
            j_id for j_id in ka_consts.KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS
            if ka_consts.KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['is_controllable']
        ]

    def _get_n_dof_gripper(self):
        return np.array(
            [1 if ka_consts.KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED' else 0
             for j_id in ka_consts.KUKA_ALLEGRO_GRIP_JOINT_ID_STATUS]
        ).sum()

    def get_arm_controlled_joint_ids(self):
        return [
            j_id for j_id in ka_consts.KUKA_ARM_JOINT_ID_STATUS
            if ka_consts.KUKA_ARM_JOINT_ID_STATUS[j_id]['status'] == 'CONTROLLED'
        ]

    def _get_controllable_joint_ids(self):
        # what joints the urdf allows us to control
        return self._get_arm_controllable_joint_ids() + self._get_gripper_controllable_joint_ids()

    def _get_rest_poses(self):
        return ku_consts.KUKA_ABOVE_OBJECT_INIT_POSITION_JOINTS

    def get_fingers(self, x):
        # Format when giving only controllable joint to robot_grasping:
        return np.array([-x for j_id in ka_consts.ALLEGRO_HAND_CONTROLLED_J_IDS])

    def _set_robot_default_state(self):
        for j_id, j_val in ka_consts.DEFAULT_JOINT_STATES.items():
            self.p.resetJointState(self.robot_id, j_id, targetValue=j_val)

    def _set_robot_manually_fixed_joint_states(self):
        for j_id, j_val in ka_consts.MANUALLY_PRESET_VALUES.items():
            self.p.resetJointState(self.robot_id, j_id, targetValue=j_val)

    def _reset_robot(self):
        self._set_robot_default_state()
        self._set_robot_manually_fixed_joint_states()




