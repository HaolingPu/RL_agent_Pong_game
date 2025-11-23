import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, RecordVideo
import torch

import ale_py

gym.register_envs(ale_py)

"""
Please read: THE ENVIRONMENT INTERFACE

In this homework you will use a version of the Pong environment adapted from
the Arcade Learning Environment (ALE). This Pong environment is built on top
of Gymnasium, a library and API for reinforcement learning that was originally
developed by OpenAI and is now open-source.

This environment exposes two primary methods for interaction:
- reset:  Resets the environment and returns the new initial state.
- step:   Applies an action and returns the next state, a reward, and two
          boolean flags that indicate whether the episode terminated or was
          truncated because it reached the maximum number of steps.

Read the implementation below to understand each function's arguments and
their return types.

For more information about the underlying libraries, see
https://gymnasium.farama.org/ and https://ale.farama.org/.
"""


class PongEnviroment:
    def __init__(self, max_steps: int | None = None, record: bool = False):
        """
        Initialize the environment.
        """
        self.env = gym.make(
            "ALE/Pong-v5",
            render_mode="rgb_array",
            max_episode_steps=max_steps,
            obs_type="ram",
        )
        self.env = FrameStackObservation(self.env, stack_size=2)
        if record:
            self.env = RecordVideo(
                self.env,
                video_folder="outputs/",
                name_prefix="video-",
                episode_trigger=lambda i: True,
                fps=10,
            )
        self.action_space = gym.spaces.Discrete(2)  # 2 possible actions: up or down
        self.observation_space = gym.spaces.Box(-1, 1, shape=(6,))

        # From https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
        self.ram_indices = dict(
            player_y=51,
            enemy_y=50,
            ball_x=49,
            ball_y=54,
        )
        self.mean = torch.tensor([90, 90, 90, 90, 0, 0])
        self.std = torch.tensor([60, 60, 60, 60, 17, 17])

        # The values are reorded to handle a binary policy (0, 1)
        # or a multiclass policy (0, 1, 2)
        self.action_map = {
            0: 3,  # Down,
            1: 2,  # Up
            2: 0,  # NOOP
        }

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, bool]:
        """
        Updates the environment based on the action taken. The action parameter
        is an integer in the range [0, ... self.action_space - 1]. Note that the
        Pong environment allows for six different actions, but we restrict to
        only two to make learning easier.

        Parameters:
            action (int): Integer representing the action taken. For Pong, this
                          is Down (0) or Up (1)

        Returns:
            (1) state : A pytorch.Tensor of size self.state_space, representing the
                        new state that the agent is in after taking its specified
                        action.
            (2) reward : A float indicating the reward received at this step.
            (3) terminated : A bool indicating whether the episode has
                             terminated; if this is True, you should reset the
                             environment and move on to the next episode.
            (4) truncated: A bool indicating if we reached the maximum number of steps
        """
        if action in self.action_map:
            action = self.action_map[action]
            state, reward, terminated, truncated, info = self.env.step(action)
            state = self._preprocess_state(state)
            terminated = terminated or reward != 0
            return (state, float(reward), terminated, truncated)
        else:
            raise ValueError(
                f"Invalid action: {action}. Action must be in {list(self.action_map.keys())}."
            )

    def reset(self, seed: int | None = None):
        state, info = self.env.reset(seed=seed)
        state = self._preprocess_state(state)
        return state

    def _preprocess_state(self, state: np.ndarray):
        player_pos = state[-1, self.ram_indices["player_y"]].item()
        enemy_pos = state[-1, self.ram_indices["enemy_y"]]
        ball_x_pos = state[-1, self.ram_indices["ball_x"]].item()
        ball_y_pos = state[-1, self.ram_indices["ball_y"]].item()
        ball_x_direction = ball_x_pos - state[0, self.ram_indices["ball_x"]].item()
        ball_y_direction = ball_y_pos - state[0, self.ram_indices["ball_y"]].item()
        processed_state = torch.tensor(
            [
                player_pos,
                enemy_pos,
                ball_x_pos,
                ball_y_pos,
                ball_x_direction,
                ball_y_direction,
            ],
            dtype=torch.float,
        )
        processed_state = (processed_state - self.mean) / self.std
        return processed_state

    def close(self, *args, **kwargs):
        self.env.close(*args, **kwargs)
        return
