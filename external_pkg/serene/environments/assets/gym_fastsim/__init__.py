import logging
from gym.envs.registration import register
from os.path import dirname, join

default_env = "assets/LS_maze_hard.xml"

logger = logging.getLogger(__name__)

register(
    id='FastsimSimpleNavigation-v0',
    entry_point='gym_fastsim.simple_nav:SimpleNavEnv',
    kwargs={"xml_env":join(dirname(__file__), default_env)}
)
