
import pdb

import numpy as np
from gym import spaces

import utils.constants as consts


class BulletSimRobot:
    def __init__(
            self,
            bullet_client,
            robot_class,
            joint_ids,
            end_effector_id,
            n_control_gripper,
            ws_radius,
            center_workspace,
            contact_ids,
            allowed_collision_pair,
            disabled_collision_pair,
            initial_state,
            disabled_obj_robot_contact_ids,
            is_there_primitive_gene,
            n_body_gripper,
            controller_type,
            change_dynamics,
            n_dof_arm
    ):

        self.change_dynamics = None  # dict containing dynamics arguments to set to the bullet simulation
        self.robot_class = None  # function that can be called to create the bullet instance corresponding to the robot
        self.controller_type = None  # str describing the type of the controller
        self.action_space = None  #  MDP space
        self.robot_id = None  # # body's id associated to the robot in the simulation

        self.joint_ids = None  # ids of controllable joints, ending with the gripper
        self.end_effector_id = None  # link id of the end effector
        self.n_control_gripper = None  # number of controlled joint belonging to the gripper
        self.n_body_gripper = None  # number of controlled joint belonging to the body (=0 for body-less manipulators)
        self.ws_radius = None  # radius of the workspace
        self.center_workspace = None  # position of the center of the workspace
        self.contact_ids = None  # robots' link ids with allowed contacts
        self.disabled_obj_robot_contact_ids = None  # list of robot joint id that cannot be touched by the object
        self.allowed_collision_pair = None  # 2D array (-1,2), list of pair of link id (int) allowed in autocollision

        self.n_joints = None  # number of joint positions
        self.n_dof_arm = None  # number of dof associated to the manipulator
        self.is_there_primitive_gene = None  # Flag triggered if the genome contains a primitive gene (for dexterous
                                             # hands, False otherwise)
        self.n_actions = None  # len of the action vector
        self.n_controllable_joints = None  # number of controllable (w.r.t. urdf) joints
        self.center_workspace_cartesian = None
        self.center_workspace_robot_frame = None
        self.initial_state = None  # robot initial state

        self._init_attributes(
            bullet_client=bullet_client,
            robot_class=robot_class,
            joint_ids=joint_ids,
            end_effector_id=end_effector_id,
            n_control_gripper=n_control_gripper,
            ws_radius=ws_radius,
            center_workspace=center_workspace,
            contact_ids=contact_ids,
            allowed_collision_pair=allowed_collision_pair,
            disabled_collision_pair=disabled_collision_pair,
            initial_state=initial_state,
            disabled_obj_robot_contact_ids=disabled_obj_robot_contact_ids,
            is_there_primitive_gene=is_there_primitive_gene,
            n_body_gripper=n_body_gripper,
            controller_type=controller_type,
            change_dynamics=change_dynamics,
            n_dof_arm=n_dof_arm
        )

    def _init_attributes(
            self,
            bullet_client,
            robot_class,
            joint_ids,
            end_effector_id,
            n_control_gripper,
            ws_radius,
            center_workspace,
            contact_ids,
            allowed_collision_pair,
            disabled_collision_pair,
            initial_state,
            disabled_obj_robot_contact_ids,
            is_there_primitive_gene,
            n_body_gripper,
            controller_type,
            change_dynamics,
            n_dof_arm
    ):

        self.robot_class = robot_class
        self.controller_type = controller_type
        self.change_dynamics = change_dynamics
        self.end_effector_id = end_effector_id
        self.n_control_gripper = n_control_gripper
        self.n_body_gripper = n_body_gripper
        self.ws_radius = ws_radius
        self.center_workspace = center_workspace
        self.contact_ids = contact_ids
        self.disabled_obj_robot_contact_ids = disabled_obj_robot_contact_ids

        self.robot_id = self.robot_class()
        self.joint_ids = self._init_joint_ids(joint_ids)
        self.allowed_collision_pair = self._init_allowed_collision_pairs(allowed_collision_pair)

        n_joints = len(self.joint_ids)
        self.n_joints = n_joints
        self.n_dof_arm = n_dof_arm
        self.is_there_primitive_gene = is_there_primitive_gene

        n_action_close_control = 1
        self.n_actions = n_joints - self.n_body_gripper - self.n_control_gripper + n_action_close_control + \
                         int(self.is_there_primitive_gene)

        self.n_controllable_joints = self.n_joints

        self.center_workspace_cartesian = self._init_center_workspace_cartesian(bullet_client=bullet_client)
        self.center_workspace_robot_frame = self._init_center_workspace_robot_frame(bullet_client=bullet_client)
        self._disable_collision_pair(bullet_client=bullet_client, disabled_collision_pair=disabled_collision_pair)

        self._init_scene_limits(bullet_client=bullet_client, n_joints=n_joints)

        self.initial_state = initial_state

        self._init_MDP_spaces()

    def _init_joint_ids(self, joint_ids):
        assert joint_ids is not None, 'joint_ids cannot be None'
        return np.array(joint_ids, dtype=int)

    def _init_allowed_collision_pairs(self, allowed_collision_pair):
        return [set(c) for c in allowed_collision_pair]

    def _init_center_workspace_cartesian(self, bullet_client):
        return np.array(
            bullet_client.getLinkState(self.robot_id, self.center_workspace)[0]
            if isinstance(self.center_workspace, int)
            else self.center_workspace
        )

    def _init_center_workspace_robot_frame(self, bullet_client):
        # Position of center_workspace in the robot frame
        return bullet_client.multiplyTransforms(
            *bullet_client.invertTransform(*bullet_client.getBasePositionAndOrientation(self.robot_id)),
            self.center_workspace_cartesian,
            [0, 0, 0, 1]
        )

    def _disable_collision_pair(self, bullet_client, disabled_collision_pair):
        for contact_point in disabled_collision_pair:
            bullet_client.setCollisionFilterPair(
                self.robot_id, self.robot_id, contact_point[0], contact_point[1], enableCollision=0
            )

    def _init_scene_limits(self, bullet_client, n_joints):
        self.lower_limits = np.zeros(n_joints)
        self.upper_limits = np.zeros(n_joints)
        self.max_force = np.zeros(n_joints)
        self.max_velocity = np.zeros(n_joints)

        for i, id in enumerate(self.joint_ids):
            self.lower_limits[i], self.upper_limits[i], self.max_force[i], self.max_velocity[i] = \
                bullet_client.getJointInfo(self.robot_id, id)[8:12]
            bullet_client.enableJointForceTorqueSensor(self.robot_id, id)

        # change dynamics
        for id, args in self.change_dynamics.items():
            if id in self.joint_ids:  # update limits if needed
                index = np.nonzero(self.joint_ids == id)[0][0]
                if 'jointLowerLimit' in args and 'jointUpperLimit' in args:
                    self.lower_limits[index] = args['jointLowerLimit']
                    self.upper_limits[index] = args['jointUpperLimit']
                if 'maxJointVelocity' in args:
                    self.max_velocity[index] = args['maxJointVelocity']
                if 'jointLimitForce' in args:
                    self.max_force[index] = args['jointLimitForce']

                bullet_client.changeDynamics(self.robot_id, linkIndex=id, **args)

        self.max_force = np.where(self.max_force <= 0, 100, self.max_force)  # replace bad values
        self.max_velocity = np.where(self.max_velocity <= 0, 1, self.max_velocity)
        self.max_acceleration = np.ones(n_joints) * 10  # set maximum acceleration for inverse dynamics

    def _init_MDP_spaces(self):
        assert self.action_space is None
        action_upper_bound = consts.ACTION_UPPER_BOUND
        action_space = spaces.Box(-action_upper_bound, action_upper_bound, shape=(self.n_actions,), dtype='float32')
        self.action_space = action_space

    def get_joint_state(self, bullet_client, position=True, normalized=True):
        """ Return (un)normalized joint positions (velocities) without the gripper"""
        if position:
            js_upper_lim = self.upper_limits[self.n_body_gripper:-self.n_control_gripper]
            js_lower_lim = self.lower_limits[self.n_body_gripper:-self.n_control_gripper]
            i_state = 0
            joint_state = np.array(
                [s[i_state] for s in bullet_client.getJointStates(
                    self.robot_id, self.joint_ids[self.n_body_gripper:-self.n_control_gripper]
                )]
            )

            if normalized:
                joint_state = 2 * (joint_state - js_upper_lim) / (js_upper_lim - js_lower_lim) + 1

            return joint_state
        else:
            i_vel = 1
            joint_vel = np.array(
                [js[i_vel] for js in bullet_client.getJointStates(
                    self.robot_id, self.joint_ids[self.n_body_gripper:-self.n_control_gripper]
                )]
            )
            if normalized:
                joint_vel = joint_vel / self.max_velocity[self.n_body_gripper:-self.n_control_gripper]
            return joint_vel

    def get_joints_poses_from_ik(self, bullet_client, pos, or_pry, normalized, arm_controllable_joint_ids, rest_poses,
                                 controllable_joint_ids):

        if isinstance(or_pry, np.ndarray):
            or_pry = or_pry.tolist()

        assert isinstance(or_pry, list) and len(or_pry) == 3
        orn = bullet_client.getQuaternionFromEuler(or_pry)

        ll = [
            bullet_client.getJointInfo(bodyUniqueId=self.robot_id, jointIndex=j_id)[8]
            for j_id in arm_controllable_joint_ids
        ]
        ul = [
            bullet_client.getJointInfo(bodyUniqueId=self.robot_id, jointIndex=j_id)[9]
            for j_id in arm_controllable_joint_ids
        ]
        jr = [5.8] * len(arm_controllable_joint_ids)  # based on pybullet official repo example

        joint_poses = bullet_client.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_id,
            pos,
            orn,
            lowerLimits=ll,
            upperLimits=ul,
            jointRanges=jr,
            restPoses=rest_poses
        )

        try:
            assert len(joint_poses) == len(controllable_joint_ids)
        except:
            pdb.set_trace()

        n_arm_jp = len(arm_controllable_joint_ids)
        arm_joint_poses = np.array(joint_poses[:n_arm_jp])

        if normalized:
            js_upper_lim = self.upper_limits[self.n_body_gripper:-self.n_control_gripper]
            js_lower_lim = self.lower_limits[self.n_body_gripper:-self.n_control_gripper]

            arm_joint_poses = 2 * (arm_joint_poses - js_upper_lim) / (js_upper_lim - js_lower_lim) + 1

        return arm_joint_poses

    def update_robot_infos(self, bullet_client, info):
        joint_states = bullet_client.getJointStates(self.robot_id, self.joint_ids)
        pos, vel = [0] * self.n_joints, [0] * self.n_joints
        iter_joint_update = zip(
            range(self.n_joints), joint_states, self.upper_limits, self.lower_limits, self.max_velocity
        )
        for i, state, u, l, v in iter_joint_update:
            pos[i] = 2 * (state[0] - u) / (u - l) + 1  # set between -1 and 1
            vel[i] = state[1] / v  # set between -1 and 1
            info['joint positions'][i], info['joint velocities'][i], jointReactionForces, \
            info['applied joint motor torques'][i] = state
            info['joint reaction forces'][i] = jointReactionForces[-1]  # get Mz

        sensor_torques = info['joint reaction forces'] / self.max_force  # scale to [-1,1]
        absolute_center = bullet_client.multiplyTransforms(
            *bullet_client.getBasePositionAndOrientation(self.robot_id),
            * self.center_workspace_robot_frame
        )  # the pose of center_workspace in the world
        invert = bullet_client.invertTransform(*absolute_center)

        # get information on gripper
        info['end effector position'], info['end effector xyzw'], _, _, _, _, info[
            'end effector linear velocity'], info['end effector angular velocity'] = bullet_client.getLinkState(
            self.robot_id, self.end_effector_id, computeLinkVelocity=True)

        end_pos, end_or = bullet_client.multiplyTransforms(
            *invert, info['end effector position'], info['end effector xyzw']
        )
        end_pos = np.array(end_pos) / self.ws_radius
        end_or = bullet_client.getMatrixFromQuaternion(end_or)[:6]
        end_lin_vel, _ = bullet_client.multiplyTransforms(
            *bullet_client.invertTransform((0, 0, 0), absolute_center[1]),
            info['end effector linear velocity'],
            (0, 0, 0, 1)
        )

        info['robot state'] = np.hstack(
            [end_pos, end_or, end_lin_vel, pos, vel, sensor_torques, ])  # robot state without the object state
        info['normalized joint pos'] = self.get_joint_state(bullet_client=bullet_client, position=True)
        return info

    def update_contact_infos(self, bullet_client, info, obj_id, plane_id, table_id):
        is_obj_initialized = obj_id is not None
        if is_obj_initialized:
            info['contact object robot'] = bullet_client.getContactPoints(bodyA=obj_id, bodyB=self.robot_id)
            info['contact object plane'] = bullet_client.getContactPoints(bodyA=obj_id, bodyB=plane_id)

        if table_id is not None and is_obj_initialized:
            info['contact object table'] = bullet_client.getContactPoints(bodyA=obj_id, bodyB=table_id)
            info['contact robot table'] = bullet_client.getContactPoints(bodyA=self.robot_id, bodyB=table_id)

        info['touch'], info['autocollision'] = False, False
        info['touch_points_obj'], info['touch_points_robot'] = [], []

        if is_obj_initialized:
            for c in info['contact object robot']:

                info['touch'] = info['touch'] or c[4] in self.contact_ids  # the object must touch the gripper

                if c[4] in self.contact_ids:
                    touch_points_on_obj = c[5]
                    touch_points_on_robot = c[6]
                    info['touch_points_obj'].append(touch_points_on_obj)
                    info['touch_points_robot'].append(touch_points_on_robot)

                if c[4] in self.disabled_obj_robot_contact_ids:
                    info['autocollision'] = True
                    break

        info['contact robot robot'] = bullet_client.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
        for c in info['contact robot robot']:
            if set(c[3:5]) not in self.allowed_collision_pair:
                #print('AUTOCOLLIDE : collision pair = ', c[3:5])
                info['autocollision'] = True
                break
        return info
