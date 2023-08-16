from gym.envs.registration import register

register(id='kuka_wsg50_grasping-v0', entry_point='gym_envs.envs:KukaWsg50Grasping', max_episode_steps=2000)
register(id='kuka_allegro_grasping-v0', entry_point='gym_envs.envs:KukaAllegroGrasping', max_episode_steps=1500)