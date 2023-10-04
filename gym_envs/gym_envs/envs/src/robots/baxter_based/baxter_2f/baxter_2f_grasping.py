
import pdb

import numpy as np
from pathlib import Path

import utils.constants as consts

from gym_envs.envs.src.robot_grasping import RobotGrasping
from gym_envs.envs.src.xacro import _process
import gym_envs
import gym_envs.envs.src.env_constants as env_consts
import gym_envs.envs.src.robots.baxter_based.baxter_consts as bx_consts
import gym_envs.envs.src.robots.baxter_based.baxter_2f.baxter_2f_consts as bx2f_consts


def generate_urdf_from_xacro(root_3d_models_robots, urdf, bx_xacro_kwargs):
    # create the file if doesn't exist (xacro to urdf conversion)
    xacro2urdf_kwargs = dict(
                    output=urdf,
                    just_deps=False,
                    xacro_ns=True,
                    verbosity=1,
                    mappings={
                        'finger': bx_xacro_kwargs['finger'],
                        "grip_slot": str(bx_xacro_kwargs['grip_slot']),
                        'tip': str(bx_xacro_kwargs['tip']) + "_tip",
                        "grasp": bx_xacro_kwargs['grasp'],
                        "limit_scale": str(bx_xacro_kwargs['limit_scale']),
                        "fixed_joints": str(bx_xacro_kwargs['fixed_joints'])
                    }
                )
    #pdb.set_trace()
    _process(
        root_3d_models_robots / bx2f_consts.BX_2_FINGERS_RELATIVE_PATH_XACRO,
        xacro2urdf_kwargs
    )


def get_bx_grip_slot(object_name):
    grip_slot = bx2f_consts.BX_GRIP_SLOT_PER_OBJECTS[object_name]
    if grip_slot is None:
        raise NotImplementedError(f'Baxter grip slot not defined for object {object_name}')
    return grip_slot


def init_urdf_baxter_2_fingers(object_name):

    fixed_joints = {'head_pan': 0}
    fixed_joints.update(bx2f_consts.BX_FIXED_JOINTS_RIGHT_ARM_DICT)
    root_3d_models_robots = \
        Path(gym_envs.__file__).resolve().parent / env_consts.GYM_ENVS_RELATIVE_PATH2ROBOTS_MODELS

    limit_scale = bx2f_consts.BX_DEFAULT_LIMIT_SCALE
    finger = bx2f_consts.BX_DEFAULT_FINGER
    tip = bx2f_consts.BX_DEFAULT_TIP
    grasp = bx2f_consts.BX_DEFAULT_GRASP
    grip_slot = get_bx_grip_slot(object_name)

    bx_setup_urdf_name = f'{finger}_{grip_slot}_{tip}_{grasp}_{np.round(limit_scale, 2)}_rightFixed.urdf'
    urdf = Path(root_3d_models_robots / bx2f_consts.BX_2_FINGERS_RELATIVE_PATH_GENERATED_URDF / bx_setup_urdf_name)

    urdf.parent.mkdir(exist_ok=True)
    if not urdf.is_file():
        bx_xacro_kwargs = {
            'finger': finger,
            'grip_slot': grip_slot,
            'tip': tip,
            'grasp': grasp,
            'limit_scale': limit_scale,
            'fixed_joints': fixed_joints,
        }
        generate_urdf_from_xacro(
            root_3d_models_robots=root_3d_models_robots, urdf=urdf, bx_xacro_kwargs=bx_xacro_kwargs
        )

    return urdf


def adjust_object_pose(kwargs, baxter_2_fingers_kwargs):
    if kwargs['object_name'] == 'ycb_mug':
        baxter_2_fingers_kwargs['object_position'] = [0, 0, -0.201]

    if kwargs['object_name'] in bx2f_consts.POSE_OFFSETED_OBJ_NAMES:
        object_position = np.array(consts.BULLET_OBJECT_DEFAULT_POSITION) + bx2f_consts.OFFSET_OBJ_POSE
        baxter_2_fingers_kwargs['object_position'] = object_position.tolist()

    #pdb.set_trace()
    return baxter_2_fingers_kwargs


class Baxter2FingersGrasping(RobotGrasping):

    def __init__(self,
                 **kwargs
                 ):

        baxter_2_fingers_kwargs = {
            'robot_class': self._load_model,
            #'object_name': object_name,
            'object_position': bx2f_consts.BX_DEFAULT_INIT_OBJECT_POSITION,
            'table_height': env_consts.TABLE_HEIGHT,
            'end_effector_id': bx2f_consts.BX_END_EFFECTOR_ID_FIXED_ARM,
            'joint_ids': bx2f_consts.BX_JOINT_IDS_FIXED_ARM,
            'n_control_gripper': self._get_n_dof_gripper(),
            'center_workspace': bx2f_consts.BX_CENTER_WORKSPACE_FIXED_ARM,
            'ws_radius': bx2f_consts.BX_WS_RADIUS,
            'contact_ids': bx2f_consts.BX_CONTACT_IDS_FIXED_ARM,
            'disabled_collision_pair': bx2f_consts.BX_DISABLED_COLLISION_PAIRS_FIXED_ARM,
            'change_dynamics': {},
            'is_there_primitive_gene': False,
        }

        baxter_2_fingers_kwargs = adjust_object_pose(kwargs=kwargs, baxter_2_fingers_kwargs=baxter_2_fingers_kwargs)

        self.object_name = kwargs['object_name']  # string describing the targeted object
        self.n_dof_arm = len(self._get_arm_controlled_joint_ids())
        self.i_action_grip_close = -1

        super().__init__(
            **baxter_2_fingers_kwargs,
            **kwargs
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

    def _get_gripper_command(self, action):
        # action = [cmd_each_joint, cmd_grip] -> cmd = [cmd_each_joint, -cmd_grip, -cmd_grip, cmd_grip, cmd_grip]
        action_grip_genome_val = action[self.i_action_grip_close]
        fingers_cmd = [action_grip_genome_val, -action_grip_genome_val]
        return fingers_cmd

    def _load_model(self):

        assert self.object_name is not None
        urdf = init_urdf_baxter_2_fingers(object_name=self.object_name)

        file_name = str(urdf)
        base_position = bx2f_consts.BX_BASE_POSITION
        base_orientation = bx2f_consts.BX_BASE_ORIENTATION
        use_fixed_base = True
        bx_flags = self.bullet_client.URDF_USE_SELF_COLLISION | self.bullet_client.URDF_MERGE_FIXED_LINKS

        robot_id = self.bullet_client.loadURDF(
            fileName=file_name,
            basePosition=base_position,
            baseOrientation=base_orientation,
            useFixedBase=use_fixed_base,
            flags=bx_flags
        )
        return robot_id

    def get_fingers(self, x):
        return np.array([x, -x])

    def _get_rest_poses(self):
        return bx2f_consts.BX_ABOVE_OBJECT_INIT_POSITION_JOINTS

    def _get_arm_controllable_joint_ids(self):
        return bx2f_consts.BX_LEFT_ARM_CONTROLLABLE_JOINT_IDS

    def _get_gripper_controllable_joint_ids(self):
        return bx2f_consts.BX_2_FINGERS_GRIP_CONTROLLABLE_JOINT_IDS

    def _get_arm_controlled_joint_ids(self):
        return bx2f_consts.BX_LEFT_ARM_CONTROLLED_JOINT_IDS

    def _get_gripper_controlled_joint_ids(self):
        return bx2f_consts.BX_2_FINGERS_GRIP_CONTROLLED_JOINT_IDS

    def _get_controllable_joint_ids(self):
        # what joints the urdf allows us to control
        return self._get_arm_controllable_joint_ids() + self._get_gripper_controllable_joint_ids()

    def _get_n_dof_gripper(self):
        return len(self._get_gripper_controlled_joint_ids())

    def _is_gripper_closed(self, action):
        return action[self.i_action_grip_close] < 0

    def _reset_robot(self):
        left = bx2f_consts.BX_UNTUCK_JOINT_POSITIONS_LEFT_DICT

        for i, v in enumerate(left.values()):
            self.bullet_client.resetJointState(self.robot_id, i, targetValue=v)




