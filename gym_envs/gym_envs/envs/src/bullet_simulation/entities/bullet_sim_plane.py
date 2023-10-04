
import utils.constants as consts


class BulletSimPlane:
    def __init__(self, bullet_client):
        self.plane_id = None  # id associated with the ground in the bullet simulation

        self._init_attributes(bullet_client=bullet_client)

    def _init_attributes(self, bullet_client):
        offset_plan_z = -1
        self.plane_id = bullet_client.loadURDF(
            consts.BULLET_PLANE_URDF_FILE_RPATH,
            [0, 0, offset_plan_z],
            useFixedBase=True
        )


