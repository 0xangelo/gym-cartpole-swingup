# pylint:disable=missing-docstring,protected-access,redefined-outer-name
import gym
import numpy as np
import pytest

import gym_cartpole_swingup as _


@pytest.fixture(
    params="CartPoleSwingUp-v0 "
    "CartPoleSwingUp-v1 "
    "TorchCartPoleSwingUp-v0 "
    "TorchCartPoleSwingUp-v1".split()
)
def env(request):
    return gym.make(request.param)


def test_init(env):
    assert hasattr(env, "observation_space")
    assert hasattr(env, "action_space")
    assert hasattr(env, "state")
    assert np.isfinite(env.action_space.high)
    assert np.isfinite(env.action_space.low)


def test_reset(env):
    obs = env.reset()
    assert obs in env.observation_space
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)
    assert isinstance(env.state, tuple)
    assert len(env.state) == 4


def test_step(env):
    env.reset()
    act = env.action_space.sample()
    next_obs, rew, done, info = env.step(act)

    assert next_obs in env.observation_space
    assert np.isscalar(rew)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert not list(info.keys())


def test_reward_fn(env):
    env.reset()
    state = env.state
    act = env.action_space.sample()
    env.reset()
    next_state = env.state
    rew = env.unwrapped._reward_fn(state, act, next_state)

    assert np.isscalar(rew)
    assert np.isfinite(rew)


def test_transition_fn(env):
    env.reset()
    state = env.state
    act = env.action_space.sample()
    next_state = env.unwrapped._transition_fn(state, act)

    assert isinstance(next_state, tuple)
    assert all(np.isfinite(s) for s in next_state)
    assert len(next_state) == 4
