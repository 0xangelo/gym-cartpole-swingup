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
while not done:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
```

![](https://i.imgur.com/Z8bLLM8.png)
