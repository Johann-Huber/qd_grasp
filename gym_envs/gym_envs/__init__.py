from gym.envs.registration import register

register(id='kuka_grasping-v0', entry_point='gym_envs.envs:KukaGrasping',max_episode_steps=2000)
register(id='kuka_iiwa_allegro-v0', entry_point='gym_envs.envs:Kuka_iiwa_allegro', max_episode_steps=1500)


