from gym.envs.registration import register

# Kuka based
register(id='kuka_wsg50_grasping-v0', entry_point='gym_envs.envs:KukaWsg50Grasping', max_episode_steps=2000)
register(id='kuka_allegro_grasping-v0', entry_point='gym_envs.envs:KukaAllegroGrasping', max_episode_steps=1500)

# Panda based
register(id='panda_2f_grasping-v0', entry_point='gym_envs.envs:FrankaEmikaPanda2Fingers', max_episode_steps=1500)

# UR5 based
register(id='ur5_sih_schunk_grasping-v0', entry_point='gym_envs.envs:UR5SihSchunk', max_episode_steps=1500)

# Baxter based
register(id='bx_2f_grasping-v0', entry_point='gym_envs.envs:Baxter2FingersGrasping', max_episode_steps=1500)


