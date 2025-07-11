import numpy as np
import cv2
from collections import deque
from gymnasium import Env, spaces, ActionWrapper
from gymnasium.utils import seeding
from mlagents_envs.environment import UnityEnvironment
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

class LimitToMoveOnlyActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Only vertical (3 options) × horizontal (3 options)
        self.vertical_dim = 3
        self.horizontal_dim = 3

        self.action_space = spaces.Discrete(self.vertical_dim * self.horizontal_dim)

    def action(self, flat_action):
        # Map flat action back to (vertical, horizontal)
        vertical = flat_action // self.horizontal_dim
        horizontal = flat_action % self.horizontal_dim
        rotation = 0  # always none

        return np.array([vertical, horizontal, rotation], dtype=np.int64)

class FlattenActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.orig_action_space = env.action_space
        self.action_space = spaces.Discrete(int(np.prod(env.action_space.nvec)))

    def action(self, flat_action):
        # Convert flat index to multidiscrete
        dims = self.orig_action_space.nvec
        md_action = []
        for dim in reversed(dims):
            md_action.append(flat_action % dim)
            flat_action = flat_action // dim
        return np.array(md_action[::-1])

    def reverse_action(self, md_action):
        # Convert multidiscrete to flat
        dims = self.orig_action_space.nvec
        flat_action = 0
        for i, a in enumerate(md_action):
            flat_action *= dims[i]
            flat_action += a
        return flat_action


class SharedObsUnityGymWrapper(Env):
    def __init__(self, unity_env, frame_stack=64, img_size=(168, 84), grayscale=True):
        self.env = UnityParallelEnv(unity_env)

        # left agent 0, right agent 1
        self.agent = self.env.possible_agents[1]       # agent to be controlled
        self.agent_other = self.env.possible_agents[0] # agent at opposite
        self.agent_obs = self.env.possible_agents[0]   # obs is only available in agent 0, always 0
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.grayscale = grayscale
        self.frames = deque(maxlen=frame_stack)
        self._np_random = None

        self.step_count = 0

        # Observation space
        base_obs = self.env.observation_spaces[self.agent_obs][0]
        c, h, w = base_obs.shape
        self._transpose = (c == 3)

        # Final obs shape after manual preprocessing
        if grayscale:
            obs_shape = (img_size[1], img_size[0], frame_stack)  # (stack, H, W)
        else:
            obs_shape = (img_size[1], img_size[0], frame_stack * c)  # (stack*C, H, W)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        self.action_space = self.env.action_spaces[self.agent]

    def _preprocess(self, obs):
        # Transpose from (C, H, W) → (H, W, C)
        if self._transpose:
            obs = obs.transpose(1, 2, 0)

        # Resize and grayscale
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)

        obs = obs.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        obs_dict = self.env.reset()
        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])

        for _ in range(self.frame_stack):
            self.frames.append(obs)

        if self.grayscale:
            stacked_obs = np.stack(list(self.frames), axis=-1)  # (H, W, stack)
        else:
            stacked_obs = np.concatenate(list(self.frames), axis=-1)

        return stacked_obs, {}  # (H, W, stack)

    def step(self, action):
        # Define square directions
        square_directions = [
            np.array([1, 0, 0], dtype=np.int32),   # right
            np.array([0, 1, 1], dtype=np.int32),   # up
            np.array([2, 0, 2], dtype=np.int32),  # left
            np.array([0, 2, 0], dtype=np.int32)   # down
        ]
        steps_per_side = 30  # Number of env steps per side of the square
        self.step_count += 1
        side = (self.step_count // steps_per_side) % 4
        other_action = square_directions[side]

        actions = {self.agent: action, self.agent_other: other_action}
        obs_dict, rewards, terminations, infos = self.env.step(actions)

        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
        self.frames.append(obs)

        if self.grayscale:
            stacked_obs = np.stack(list(self.frames), axis=-1)  # (H, W, stack)
        else:
            stacked_obs = np.concatenate(list(self.frames), axis=-1)

        bonus = 0
        if np.array_equal(action, np.array([0, 2, 0])):
            decay_factor = max(0, 1 - self.step_count / 10000)  # Linearly decay to 0 over 1000 steps
            bonus += 0.5 * decay_factor  # Initial bonus is 0.1

        bonus -= 5*rewards[self.agent_other]

        if (rewards[self.agent] + rewards[self.agent_other]) > 0:
            print ("Rewards: ", rewards[self.agent] + bonus, rewards[self.agent_other])
            print("Action:", action)

        

        return stacked_obs, rewards[self.agent] + bonus, terminations[self.agent], False, infos[self.agent]

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # typically 4 for stacked frames

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the output size of CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            sample_output = self.cnn(sample_input)
            cnn_output_dim = sample_output.shape[1]

        # Final linear layer to get to desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
