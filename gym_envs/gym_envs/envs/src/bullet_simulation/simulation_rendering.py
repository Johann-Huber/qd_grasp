
import utils.constants as consts
import gym_envs.envs.src.env_constants as env_consts


class SimulationRendering:

    def __init__(self, bullet_client, display, table_label):

        self._display = None  # flag that triggers rendering
        self._camera = None  # dict containing bullet camera parameters

        self._init_attributes(bullet_client=bullet_client, display=display, table_label=table_label)

    @property
    def display(self):
        return self._display

    @property
    def camera(self):
        return self._camera

    @display.setter
    def display(self, val):
        self._display = val

    def _init_attributes(self, bullet_client, display, table_label):

        self._display = display
        self._camera = self._init_camera(display=display, bullet_client=bullet_client)

        if self._display:
            self._init_display(bullet_client=bullet_client, table_label=table_label)

    def _init_camera(self, display, bullet_client):
        cam = consts.CAMERA_DEFAULT_PARAMETERS

        cam['width'] = cam.get('width', 1024)
        cam['height'] = cam.get('height', 1024)
        cam['target'] = cam.get('target', (0, 0, 0))
        cam['distance'] = cam.get('distance', 1)
        cam['yaw'] = cam.get('yaw', 180)
        cam['pitch'] = cam.get('pitch', 0)
        cam['viewMatrix'] = bullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam['target'],
            distance=cam['distance'],
            yaw=cam['yaw'],
            pitch=cam['pitch'],
            roll=cam.get('roll', 0),
            upAxisIndex=cam.get('upAxisIndex', 2),
        )
        cam['projectionMatrix'] = bullet_client.computeProjectionMatrixFOV(
            fov=cam.get('fov', 90),
            aspect=cam['width'] / cam['height'],
            nearVal=cam.get('nearVal', 0.1),
            farVal=cam.get('farVal', 10),
        )
        cam['renderer'] = bullet_client.ER_BULLET_HARDWARE_OPENGL if display else bullet_client.ER_TINY_RENDERER

        return cam

    def _init_display(self, bullet_client, table_label):
        assert self.display

        bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_RENDERING, 1)
        bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_GUI, 0)

        self.reset_camera_pose_for_specific_table(bullet_client=bullet_client, table_label=table_label)

    def reset_camera_pose_for_specific_table(self, bullet_client, table_label):
        if table_label == env_consts.TableLabel.STANDARD_TABLE:
            bullet_client.resetDebugVisualizerCamera(
                cameraDistance=1, cameraYaw=180, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.1]
            )
        elif table_label == env_consts.TableLabel.UR5_TABLE:
            bullet_client.resetDebugVisualizerCamera(
                cameraDistance=1.3, cameraYaw=180, cameraPitch=-20, cameraTargetPosition=[0.7, 0.45, -0.2]
            )
        else:
            raise RuntimeError(f'Missing table link for {table_label}')

