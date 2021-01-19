# pylint:disable=missing-module-docstring
import importlib

import poetry_version
from gym.envs.registration import register

__author__ = """Ângelo Gregório Lovatto"""
__email__ = "angelolovatto@gmail.com"
__version__ = poetry_version.extract(source_file=__file__)


register(
    id="CartPoleSwingUp-v0",
    entry_point="gym_cartpole_swingup.envs.cartpole_swingup:CartPoleSwingUpV0",
    max_episode_steps=500,
)

register(
    id="CartPoleSwingUp-v1",
    entry_point="gym_cartpole_swingup.envs.cartpole_swingup:CartPoleSwingUpV1",
    max_episode_steps=500,
)


if importlib.util.find_spec("torch"):
    register(
        id="TorchCartPoleSwingUp-v0",
        entry_point="gym_cartpole_swingup.envs.torch_cartpole_swingup:"
        "TorchCartPoleSwingUpV0",
        max_episode_steps=500,
    )

    register(
        id="TorchCartPoleSwingUp-v1",
        entry_point="gym_cartpole_swingup.envs.torch_cartpole_swingup:"
        "TorchCartPoleSwingUpV1",
        max_episode_steps=500,
    )
