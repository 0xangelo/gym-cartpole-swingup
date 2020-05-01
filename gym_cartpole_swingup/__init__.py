# pylint:disable=missing-module-docstring
from gym.envs.registration import register

register(
    id="CartPoleSwingUp-v0",
    entry_point="gym_cartpole_swingup.envs:CartPoleSwingUpV0",
    max_episode_steps=500,
)

register(
    id="CartPoleSwingUp-v1",
    entry_point="gym_cartpole_swingup.envs:CartPoleSwingUpV1",
    max_episode_steps=500,
)
