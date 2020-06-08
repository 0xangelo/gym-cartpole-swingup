"""
Cart pole swing-up: modified version of:
https://github.com/hardmaru/estool/blob/master/custom_envs/cartpole_swingup.py
"""
from dataclasses import dataclass, field
from collections import namedtuple

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


@dataclass(frozen=True)
class CartParams:
    """Parameters defining the Cart."""

    width: float = 1 / 3
    height: float = 1 / 6
    mass: float = 0.5


@dataclass(frozen=True)
class PoleParams:
    """Parameters defining the Pole."""

    width: float = 0.05
    length: float = 0.6
    mass: float = 0.5


@dataclass
class CartPoleSwingUpParams:  # pylint: disable=no-member,too-many-instance-attributes
    """Parameters for physics simulation."""

    gravity: float = 9.82
    forcemag: float = 10.0
    deltat: float = 0.01
    friction: float = 0.1
    x_threshold: float = 2.4
    cart: CartParams = field(default_factory=CartParams)
    pole: PoleParams = field(default_factory=PoleParams)
    masstotal: float = field(init=False)
    mpl: float = field(init=False)

    def __post_init__(self):
        self.masstotal = self.cart.mass + self.pole.mass
        self.mpl = self.pole.mass * self.pole.length


State = namedtuple("State", "x_pos x_dot theta theta_dot")


class CartPoleSwingUpEnv(gym.Env):
    """
    Description:
       A pole is attached by an un-actuated joint to a cart, which moves along a track.
       Unlike CartPoleEnv, friction is taken into account in the physics calculations.
       The pendulum starts (pointing down) upside down, and the goal is to swing it up
       and keep it upright by increasing and reducing the cart's velocity.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        high = np.array([1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=-high, high=high)
        high = np.array([np.finfo(np.float32).max] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high)
        self.params = CartPoleSwingUpParams()

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        state = self.state
        # Valid action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state = next_state = self._transition_fn(self.state, action)
        next_obs = self._get_obs(next_state)
        reward = self._reward_fn(state, action, next_state)
        done = self._terminal(next_state)

        return next_obs, reward, done, {}

    def reset(self):
        self.state = State(
            *self.np_random.normal(
                loc=np.array([0.0, 0.0, np.pi, 0.0]),
                scale=np.array([0.2, 0.2, 0.2, 0.2]),
            ).astype(np.float32)
        )
        return self._get_obs(self.state)

    @staticmethod
    def _reward_fn(state, action, next_state):  # pylint: disable=unused-argument
        raise NotImplementedError

    def _terminal(self, state):
        return bool(abs(state.x_pos) > self.params.x_threshold)

    def _transition_fn(self, state, action):
        # pylint: disable=no-member
        action = action[0] * self.params.forcemag

        sin_theta = np.sin(state.theta)
        cos_theta = np.cos(state.theta)

        xdot_update = (
            -2 * self.params.mpl * (state.theta_dot ** 2) * sin_theta
            + 3 * self.params.pole.mass * self.params.gravity * sin_theta * cos_theta
            + 4 * action
            - 4 * self.params.friction * state.x_dot
        ) / (4 * self.params.masstotal - 3 * self.params.pole.mass * cos_theta ** 2)
        thetadot_update = (
            -3 * self.params.mpl * (state.theta_dot ** 2) * sin_theta * cos_theta
            + 6 * self.params.masstotal * self.params.gravity * sin_theta
            + 6 * (action - self.params.friction * state.x_dot) * cos_theta
        ) / (
            4 * self.params.pole.length * self.params.masstotal
            - 3 * self.params.mpl * cos_theta ** 2
        )

        delta_t = self.params.deltat
        return State(
            x_pos=state.x_pos + state.x_dot * delta_t,
            theta=state.theta + state.theta_dot * delta_t,
            x_dot=state.x_dot + xdot_update * delta_t,
            theta_dot=state.theta_dot + thetadot_update * delta_t,
        )

    @staticmethod
    def _get_obs(state):
        x_pos, x_dot, theta, theta_dot = state
        return np.array(
            [x_pos, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32
        )

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = CartPoleSwingUpViewer(
                self.params.cart, self.params.pole, world_width=5
            )

        if self.state is None:
            return None

        self.viewer.update(self.state, self.params.pole)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CartPoleSwingUpV0(CartPoleSwingUpEnv):
    """CartPoleSwingUp with cosine reward."""

    @staticmethod
    def _reward_fn(state, action, next_state):
        return np.cos(next_state.theta, dtype=np.float32)


class CartPoleSwingUpV1(CartPoleSwingUpEnv):
    """CartPoleSwingUp with strictly positive reward."""

    @staticmethod
    def _reward_fn(state, action, next_state):
        return (1 + np.cos(next_state.theta, dtype=np.float32)) / 2


Screen = namedtuple("Screen", "width height")


class CartPoleSwingUpViewer:
    """Class that encapsulates all the variables and objectecs needed
       to render a CartPoleSwingUpEnv. It handles all the initialization
       and updating of each object on screen and handles calls to the underlying
       gym.envs.classic_control.rendering.Viewer instance.
    """

    screen = Screen(width=600, height=400)

    def __init__(self, cart, pole, world_width):
        # pylint:disable=import-outside-toplevel
        from gym.envs.classic_control import rendering

        # pylint:enable=import-outside-toplevel

        self.world_width = world_width
        screen = self.screen
        scale = screen.width / self.world_width
        cartwidth, cartheight = scale * cart.width, scale * cart.height
        polewidth, polelength = scale * pole.width, scale * pole.length
        self.viewer = rendering.Viewer(screen.width, screen.height)
        self.transforms = {
            "cart": rendering.Transform(),
            "pole": rendering.Transform(translation=(0, 0)),
            "pole_bob": rendering.Transform(),
            "wheel_l": rendering.Transform(
                translation=(-cartwidth / 2, -cartheight / 2)
            ),
            "wheel_r": rendering.Transform(
                translation=(cartwidth / 2, -cartheight / 2)
            ),
        }

        self._init_track(rendering, cartheight)
        self._init_cart(rendering, cartwidth, cartheight)
        self._init_wheels(rendering, cartheight)
        self._init_pole(rendering, polewidth, polelength)
        self._init_axle(rendering, polewidth)
        # Make another circle on the top of the pole
        self._init_pole_bob(rendering, polewidth)

    def _init_track(self, rendering, cartheight):
        screen = self.screen
        carty = screen.height / 2
        track_height = carty - cartheight / 2 - cartheight / 4
        track = rendering.Line((0, track_height), (screen.width, track_height))
        track.set_color(0, 0, 0)
        self.viewer.add_geom(track)

    def _init_cart(self, rendering, cartwidth, cartheight):
        lef, rig, top, bot = (
            -cartwidth / 2,
            cartwidth / 2,
            cartheight / 2,
            -cartheight / 2,
        )
        cart = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
        cart.add_attr(self.transforms["cart"])
        cart.set_color(1, 0, 0)
        self.viewer.add_geom(cart)

    def _init_pole(self, rendering, polewidth, polelength):
        lef, rig, top, bot = (
            -polewidth / 2,
            polewidth / 2,
            polelength - polewidth / 2,
            -polewidth / 2,
        )
        pole = rendering.FilledPolygon([(lef, bot), (lef, top), (rig, top), (rig, bot)])
        pole.set_color(0, 0, 1)
        pole.add_attr(self.transforms["pole"])
        pole.add_attr(self.transforms["cart"])
        self.viewer.add_geom(pole)

    def _init_axle(self, rendering, polewidth):
        axle = rendering.make_circle(polewidth / 2)
        axle.add_attr(self.transforms["pole"])
        axle.add_attr(self.transforms["cart"])
        axle.set_color(0.1, 1, 1)
        self.viewer.add_geom(axle)

    def _init_pole_bob(self, rendering, polewidth):
        pole_bob = rendering.make_circle(polewidth / 2)
        pole_bob.add_attr(self.transforms["pole_bob"])
        pole_bob.add_attr(self.transforms["pole"])
        pole_bob.add_attr(self.transforms["cart"])
        pole_bob.set_color(0, 0, 0)
        self.viewer.add_geom(pole_bob)

    def _init_wheels(self, rendering, cartheight):
        wheel_l = rendering.make_circle(cartheight / 4)
        wheel_r = rendering.make_circle(cartheight / 4)
        wheel_l.add_attr(self.transforms["wheel_l"])
        wheel_l.add_attr(self.transforms["cart"])
        wheel_r.add_attr(self.transforms["wheel_r"])
        wheel_r.add_attr(self.transforms["cart"])
        wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
        wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
        self.viewer.add_geom(wheel_l)
        self.viewer.add_geom(wheel_r)

    def update(self, state, pole):
        """Updates the positions of the objects on screen"""
        screen = self.screen
        scale = screen.width / self.world_width

        cartx = state.x_pos * scale + screen.width / 2.0  # MIDDLE OF CART
        carty = screen.height / 2
        self.transforms["cart"].set_translation(cartx, carty)
        self.transforms["pole"].set_rotation(state.theta)
        self.transforms["pole_bob"].set_translation(
            -pole.length * np.sin(state.theta), pole.length * np.cos(state.theta)
        )

    def render(self, *args, **kwargs):
        """Forwards the call to the underlying Viewer instance"""
        return self.viewer.render(*args, **kwargs)

    def close(self):
        """Closes the underlying Viewer instance"""
        self.viewer.close()
