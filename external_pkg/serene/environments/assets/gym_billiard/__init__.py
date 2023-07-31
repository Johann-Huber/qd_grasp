from gym.envs.registration import register

register(
    id='Billiard-v0',
    entry_point='gym_billiard.envs:BilliardEnv',
    # timestep_limit=1000,
)
register(
    id='BilliardHard-v0',
    entry_point='gym_billiard.envs:BilliardHardEnv',
    # timestep_limit=1000,
)

register(
    id='Curling-v0',
    entry_point='gym_billiard.envs:Curling',
    # timestep_limit=1000,
)