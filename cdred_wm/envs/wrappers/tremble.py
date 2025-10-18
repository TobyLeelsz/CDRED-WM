from collections import defaultdict

import gym
import numpy as np
import torch


class TremblingHandWrapper(gym.Wrapper):
    """
    Add noise to action space according to `p_tremble`.
    """

    def __init__(self, env: gym.Env, p_tremble: float = 0.01):
        super().__init__(env)
        self.env = env
        self.p_tremble = p_tremble
        self.rng = np.random.default_rng()

        # env_name = (
        #     env.unwrapped.spec.id
        #     if hasattr(env.unwrapped.spec, "id")
        #     else env.env_name()
        # )
        # if float(p_tremble) != 0.0:
        #     cprint(
        #         f"Shaking {env_name} with probability {self.p_tremble}", attrs=["bold"]
        #     )

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if self.rng.random() < self.p_tremble:
            action = self.env.action_space.sample()
        return self.env.step(action)