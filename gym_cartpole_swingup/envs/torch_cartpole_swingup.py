# pylint:disable=missing-module-docstring,import-error
import torch

from .cartpole_swingup import CartPoleSwingUpEnv, State


class TorchCartPoleSwingUpEnv(CartPoleSwingUpEnv):
    """
    CartPoleSwingUp task that exposes differentiable reward and transition functions.
    """

    # pylint:disable=too-few-public-methods

    def _transition_fn(self, state, action):
        action = torch.as_tensor(action).float()
        state = torch.as_tensor(state)
        expanded_next_state, _ = self.transition_fn(self.expand_state(state), action)
        next_state = self.shrink_state(expanded_next_state)
        return State(*next_state.numpy())

    def _reward_fn(self, state, action, next_state):
        action = torch.as_tensor(action).float()
        state = torch.as_tensor(state)
        next_state = torch.as_tensor(next_state)
        reward = self.reward_fn(
            self.expand_state(state), action, self.expand_state(next_state)
        )
        return reward.item()

    def _terminal(self, state):
        state = torch.as_tensor(state)
        return self.terminal(state).item()

    def terminal(self, state):
        """Return a batched tensor indicating which states are terminal."""
        return (state[..., 0] < -self.params.x_threshold) | (
            state[..., 0] > self.params.x_threshold
        )

    @staticmethod
    def expand_state(state):
        """Return state with theta broken down into sin and cos."""
        return torch.cat(
            [
                state[..., :2],
                state[..., 2:3].cos(),
                state[..., 2:3].sin(),
                state[..., 3:],
            ],
            dim=-1,
        )

    @staticmethod
    def shrink_state(state):
        """Return state with sin and cos combined into theta."""
        return torch.cat(
            [state[..., :2], torch.atan2(state[3:4], state[..., 2:3]), state[..., 4:]],
            dim=-1,
        )

    def transition_fn(self, state, action, sample_shape=()):
        """Compute the next state and its log-probability.

        Accepts a `sample_shape` argument to sample multiple next states.
        """
        # pylint: disable=no-member,unused-argument
        action = action[..., 0] * self.params.forcemag

        xdot_update = self._calculate_xdot_update(state, action)
        thetadot_update = self._calculate_thetadot_update(state, action)

        delta_t = self.params.deltat
        new_x = state[..., 0] + state[..., 1] * delta_t
        new_xdot = state[..., 1] + xdot_update * delta_t
        new_costheta, new_sintheta = self._calculate_theta_update(state, delta_t)
        new_thetadot = state[..., 4] + thetadot_update * delta_t

        next_state = torch.stack(
            [new_x, new_xdot, new_costheta, new_sintheta, new_thetadot], dim=-1
        )
        return next_state.expand(torch.Size(sample_shape) + state.shape), None

    def _calculate_xdot_update(self, state, action):
        # pylint: disable=no-member
        x_dot = state[..., 1]
        theta_dot = state[..., 4]
        cos_theta = state[..., 2]
        sin_theta = state[..., 3]
        return (
            -2 * self.params.mpl * (theta_dot ** 2) * sin_theta
            + 3 * self.params.pole.mass * self.params.gravity * sin_theta * cos_theta
            + 4 * action
            - 4 * self.params.friction * x_dot
        ) / (4 * self.params.masstotal - 3 * self.params.pole.mass * cos_theta ** 2)

    def _calculate_thetadot_update(self, state, action):
        # pylint: disable=no-member
        x_dot = state[..., 1]
        theta_dot = state[..., 4]
        cos_theta = state[..., 2]
        sin_theta = state[..., 3]
        return (
            -3 * self.params.mpl * (theta_dot ** 2) * sin_theta * cos_theta
            + 6 * self.params.masstotal * self.params.gravity * sin_theta
            + 6 * (action - self.params.friction * x_dot) * cos_theta
        ) / (
            4 * self.params.pole.length * self.params.masstotal
            - 3 * self.params.mpl * cos_theta ** 2
        )

    @staticmethod
    def _calculate_theta_update(state, delta_t):
        cos_theta = state[..., 2]
        sin_theta = state[..., 3]
        sin_theta_dot = torch.sin(state[..., 4] * delta_t)
        cos_theta_dot = torch.cos(state[..., 4] * delta_t)
        new_sintheta = sin_theta * cos_theta_dot + cos_theta * sin_theta_dot
        new_costheta = cos_theta * cos_theta_dot - sin_theta * sin_theta_dot
        return new_costheta, new_sintheta

    @staticmethod
    def reward_fn(state, action, next_state):  # pylint: disable=unused-argument
        """Return the batched reward for the batched transition."""
        raise NotImplementedError


class TorchCartPoleSwingUpV0(TorchCartPoleSwingUpEnv):
    # pylint:disable=missing-docstring
    @staticmethod
    def reward_fn(state, action, next_state):  # pylint: disable=unused-argument
        return next_state[..., 2]


class TorchCartPoleSwingUpV1(TorchCartPoleSwingUpEnv):
    # pylint:disable=missing-docstring
    @staticmethod
    def reward_fn(state, action, next_state):  # pylint: disable=unused-argument
        return (1 + next_state[..., 2]) / 2
