import pdb

import numpy as np
from pathlib import Path


class BulletSimObject:
    def __init__(self, bullet_client, xyz_pose, xyzw_orient, name):

        self.object_position = None  # object position (in m)
        self.object_xyzw = None  # object orientation (in quaternions)
        self.frictions = None  # dict containing friction parameters
        self._initial_stabilized_object_pos = None  # (pos, qua) of the stabilized object on the table
        self.object_name = None  # striped object name
        self.obj_id = None  # id associated with the object in the bullet simulation

        self._init_attributes(
            bullet_client=bullet_client,
            xyz_pose=xyz_pose,
            xyzw_orient=xyzw_orient,
            name=name
        )

    def _init_attributes(self, bullet_client, xyz_pose, xyzw_orient, name):

        self.object_position = xyz_pose
        self.object_xyzw = xyzw_orient
        self._initial_stabilized_object_pos = None  # must be set after simulation stabilization

        if name is None:
            print('Warning : no given object.')
            return

        self.object_name = self._init_object_name(name)
        self._load_object_bullet(bullet_client=bullet_client)
        self.frictions = self._init_frictions(bullet_client=bullet_client)

    def _load_object_bullet(self, bullet_client):
        self.load_object(bullet_client=bullet_client, obj=self.object_name)

    def _init_object_name(self, object_name):
        return object_name.strip() if object_name is not None else None

    def load_object(self, bullet_client, obj=None):
        pos = np.array(self.object_position)

        assert isinstance(obj, str)

        urdf = Path(__file__).parent.parent.parent.parent/"3d_models/objects"/obj/f"{obj}.urdf"
        if not urdf.exists():
            raise ValueError(str(urdf) + " doesn't exist")

        try:
            obj_to_grab_id = bullet_client.loadURDF(str(urdf), pos, self.object_xyzw, useMaximalCoordinates=True)
        except bullet_client.error as e:
            raise bullet_client.error(f"{e}: " + str(urdf))

        bullet_client.changeDynamics(
            obj_to_grab_id, -1, spinningFriction=1e-2, rollingFriction=1e-3, lateralFriction=0.5
        )

        self.obj_id = obj_to_grab_id

    def _init_frictions(self, bullet_client):
        dynamicsInfo = bullet_client.getDynamicsInfo(self.obj_id, -1)  # save intial friction coefficient of the object
        frictions = {'lateral': dynamicsInfo[1], 'rolling': dynamicsInfo[6], 'spinning': dynamicsInfo[7]}
        return frictions

    def update_infos(self, bullet_client, info):
        is_obj_initialized = self.obj_id is not None
        if is_obj_initialized:
            obj_pose = bullet_client.getBasePositionAndOrientation(self.obj_id)
            obj_vel = bullet_client.getBaseVelocity(self.obj_id)
            info['object position'], info['object xyzw'] = obj_pose
            info['object linear velocity'], info['object angular velocity'] = obj_vel
        return info

    def reset_object_pose(self, bullet_client, obj_init_state_offset):

        assert obj_init_state_offset is not None
        assert isinstance(obj_init_state_offset, dict)
        assert 'init_pos_offset' in obj_init_state_offset
        assert 'init_orient_offset_euler' in obj_init_state_offset

        init_pos_offset = obj_init_state_offset['init_pos_offset']
        init_orient_offset_euler = obj_init_state_offset['init_orient_offset_euler']
        #pdb.set_trace()
        #pos = self.object_position
        #qua = self.object_xyzw
        pos, qua = bullet_client.getBasePositionAndOrientation(self.obj_id)

        # Offset on position
        assert len(init_pos_offset) == 2
        pos = list(pos)
        pos[0] += init_pos_offset[0]  # x offset
        pos[1] += init_pos_offset[1]  # y offset

        # Offset on orientation
        assert isinstance(init_orient_offset_euler, np.ndarray)
        assert len(init_orient_offset_euler) == 3

        end_eff_euler_orient = np.array(bullet_client.getEulerFromQuaternion(qua))
        end_eff_euler_orient += init_orient_offset_euler
        qua = bullet_client.getQuaternionFromEuler(end_eff_euler_orient)

        pos, qua = tuple(pos), tuple(qua)
        bullet_client.resetBasePositionAndOrientation(self.obj_id, pos, qua)




