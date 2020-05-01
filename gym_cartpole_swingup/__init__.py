# pylint:disable=missing-module-docstring
from gym.envs.registration import register

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


try:
    import torch as _

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
except ImportError:
    pass
