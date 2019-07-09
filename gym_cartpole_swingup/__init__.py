# pylint: disable=missing-docstring
from gym.envs.registration import register

register(
    id="CartPoleSwingUp-v0",
    entry_point="gym_cartpole_swingup.envs:CartPoleSwingUpEnv",
    max_episode_steps=500,
)
