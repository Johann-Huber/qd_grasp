# Created by Giuseppe Paolo 
# Date: 22/09/2020
from gym.envs.registration import register

register(
    id='AntObstacles-v0',
    entry_point='ant_maze.ant_maze:AntObstaclesBigEnv',
)