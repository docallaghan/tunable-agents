from gym.envs.registration import register

register(
    id='wolfpack-custom-v0',
    entry_point='gym_wolfpack_custom.envs:WolfpackCustomEnv',
)

register(
    id='mo-wolfpack-custom-v0',
    entry_point='gym_wolfpack_custom.envs:MOWolfpackCustomEnv',
)
