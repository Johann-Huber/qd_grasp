# Created by Giuseppe Paolo 
# Date: 27/08/2020
from gym.envs.registration import register

register(
  id='CollectBall-v0',
  entry_point='gym_collectball.envs:CollectBall'
)