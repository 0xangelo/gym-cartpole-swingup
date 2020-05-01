# gym-cartpole-swingup
A simple, continuous-control environment for OpenAI Gym

## Installation
```bash
pip install gym-cartpole-swingup
```

## Usage example
```python
# coding: utf-8
import gym
import gym_cartpole_swingup

# Could be one of:
# CartPoleSwingUp-v0, CartPoleSwingUp-v1
# If you have PyTorch installed:
# TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
env = gym.make("CartPoleSwingUp-v0")
done = False

while not done:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
```

![](https://i.imgur.com/Z8bLLM8.png)
