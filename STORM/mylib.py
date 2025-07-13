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
from stable_baselines3 import PPO

import traceback

class FlatToMultiDiscreteWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.orig_action_space = env.action_space
        self.action_space = spaces.Discrete(int(np.prod(env.action_space.nvec)))

    def action(self, flat_action):
        vertical = flat_action // 9
        horizontal = (flat_action % 9) // 3
        rotation = flat_action % 3
        return np.array([vertical, horizontal, rotation], dtype=np.int64)
    
    def multidiscrete_to_flat(action):
        vertical, horizontal, rotation = action
        return vertical * 9 + horizontal * 3 + rotation
    
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
    def __init__(self, unity_env, frame_stack=16, img_size=(168, 84), grayscale=True, ppo_agents=None):
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

        self.ball_touched = False
        self.agent_touched = None

        self.frames2 = deque(maxlen=8)

        self.ball_coords = None

        self.ppo_agents = ppo_agents or []
        self.scripted_opponents = ["square", "dash"]
        combined = self.ppo_agents + self.scripted_opponents
        self.opponents = combined if combined else None

        self.opponent_policy = None
        self.opponent_type = "attack"

        self.is_serving = True

        self._select_opponent()

        # >>> zy's reward parameters
        self.prev_potential = 0.5
        self.shaping_lambda = 0.3
        self.anneal_every = 50_000
        self.anneal_factor = 0.95
        # zy's reward parameters <<<


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

    def _select_opponent(self):
        self.opponent_policy = np.random.choice(self.opponents)

        print(f"Switched opponent to: {self.opponent_policy}")

    def _preprocess(self, obs):
        # Transpose from (C, H, W) → (H, W, C)
        if self._transpose:
            obs = obs.transpose(1, 2, 0)

        # >>> BT codes
        obs = (obs * 255).astype(np.uint8)
        agent_border_mask_low = np.array([235,235,235], dtype=np.uint8)
        agent_border_mask_high = np.array([255,255,255], dtype=np.uint8)
        agent_border_mask = cv2.inRange(obs, agent_border_mask_low, agent_border_mask_high)
        agent_mask = self._remove_border(agent_border_mask)

        ball_mask_low = np.array([220,223,31], dtype=np.uint8)
        ball_mask_high = np.array([250,250,50], dtype=np.uint8)
        ball_mask = cv2.inRange(obs, ball_mask_low, ball_mask_high)
        self.ball_coords = self._get_ball_coords(ball_mask)

        kernel = np.ones((5, 5), np.uint8)
        dilated_ball_mask = cv2.dilate(ball_mask, kernel, iterations=1)
        self.ball_touched, self.agent_touched = self._get_ball_touched(dilated_ball_mask, agent_mask, self.ball_coords)
        
        final_obs = agent_mask | dilated_ball_mask

        obs = final_obs.astype(np.float32) / 255.0

        # BT codes <<<

        # Resize and grayscale
        obs = cv2.resize(final_obs, self.img_size, interpolation=cv2.INTER_AREA)

        if self.opponents is not None:
            flipped_obs = cv2.flip(obs, 1)
            return obs, flipped_obs

        # if self.grayscale:
        #     obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)

        # obs = obs.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return obs

    def _get_ball_coords(self, obs):
        ball_coords = np.unravel_index(np.argmax(obs, axis=None), obs.shape)    # first occurence of ball pixel (top left of ball)
        _ = 2   # translation
        return ball_coords[0]+_, ball_coords[1]+_   # from top left to roughly the centre
    
    def _remove_border(self, obs):
        border_coords = np.array([
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
        ])
        # print(border_coords)
        obs[border_coords] = 0
        # plt.imshow(obs)
        # plt.show()
        return obs
    
    def _get_ball_touched(self, ball_obs, agent_obs, ball_coords):
        ball_touched = False
        # print(np.argmax(ball_obs & agent_obs))
        if np.argmax(ball_obs & agent_obs) > 0:
            ball_touched = True
            # plt.imshow(ball_obs & agent_obs)
            # plt.show()
        if not ball_touched:
            return ball_touched, None   # no agent has touched the ball
        else:
            return ball_touched, self.agent if ball_coords[1] > ball_obs.shape[1]//2 else self.agent_other
        
    def _get_ball_side(self, ball_coords, obs):
        return self.agent if ball_coords[1] > obs.shape[1]//2 else self.agent_other

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        obs_dict = self.env.reset()

        if self.opponents is not None:
            obs, flipped_obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
            for _ in range(self.frame_stack):
                self.frames.append(obs)
                self.frames2.append(flipped_obs)
        else:
            obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
            for _ in range(self.frame_stack):
                self.frames.append(obs)

        if self.grayscale:
            stacked_obs = np.stack(list(self.frames), axis=-1)  # (H, W, stack)
        else:
            stacked_obs = np.concatenate(list(self.frames), axis=-1)


        unmasked_obs = self._ori_preprocess(obs_dict[self.agent_obs]['observation'][0])
        return stacked_obs, {"unmasked_obs": unmasked_obs}  # (H, W, stack)

    def step_opponent(self):
        # Agent 0 action
        if self.opponent_type == "attack":
            if isinstance(self.opponent_policy, str):  # Scripted opponent
                if self.opponent_policy == "square":
                    steps_per_side = 30
                    side = (self.step_count // steps_per_side) % 4
                    square_directions = [
                        np.array([1, 0, 0], dtype=np.int32),
                        np.array([0, 1, 1], dtype=np.int32),
                        np.array([2, 0, 2], dtype=np.int32),
                        np.array([0, 2, 0], dtype=np.int32)
                    ]
                    other_action = square_directions[side]
                elif self.opponent_policy == "dash":
                    other_action = np.array([0, 1, 0], dtype=np.int32)
                    other_action[2] = np.random.randint(3)

                else:
                    raise ValueError(f"Unknown scripted opponent: {self.opponent_policy}")
            elif isinstance(self.opponent_policy, dict):  # PPO agent
                agent = self.opponent_policy["agent"]
                required_stack = self.opponent_policy["stack"]
                
                if len(self.frames2) < required_stack:
                    other_action = np.array([0, 1, 0], dtype=np.int32)  # fallback
                else:
                    # PPO agent
                    invert = {
                        0:0,
                        1:2,
                        2:1
                    }
                    agent = self.opponent_policy["agent"]
                    required_stack = self.opponent_policy["stack"]

                    frames_np = np.stack(list(self.frames2)[-required_stack:], axis=0)  # [stack, H, W]

                    if required_stack == 4:
                        target_size = (64, 148)  # (H, W)

                        ppo_input = np.stack([
                            cv2.resize(f, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
                            for f in frames_np
                        ], axis=0)  # [stack, 64, 148]

                        other_action, _states = agent.predict(ppo_input)
                        
                        other_action[1] = invert[other_action[1]]
                        other_action = np.append(other_action, 0)

                    else:
                        ppo_input = frames_np

                        # Run prediction
                        other_action, _states = agent.predict(ppo_input)
                        
                        other_action[1] = invert[other_action[1]]
                        other_action[2] = invert[other_action[2]]
            else: 
                print(f"Unknown opponent policy: {self.opponent_policy}")
        else:
            other_action = np.array([0, 0, 0], dtype=np.int32)  
            other_action[0] = np.random.randint(3)  # random action
            other_action[2] = np.random.randint(3)  # random action

        return other_action
                
    def _ori_preprocess(self, obs):
        # Transpose from (C, H, W) → (H, W, C)
        if self._transpose:
            obs = obs.transpose(1, 2, 0)

        # Resize and grayscale
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
            obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        else:
            obs = obs.transpose(2, 0, 1)  # (C, H, W)

        obs = obs.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        return obs
    
    def step(self, action):
        try:
            other_action = self.step_opponent()

            actions = {self.agent: action, self.agent_other: other_action}
            obs_dict, rewards, terminations, infos = self.env.step(actions)
            gamestate = obs_dict[self.agent_obs]['observation'][1]
            gamestate = int(gamestate[0].item())

            self.step_count += 1

            if self.opponents is not None:
                obs, flipped_obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
                self.frames.append(obs)
                self.frames2.append(flipped_obs)
            else:
                obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
                self.frames.append(obs)

            if self.grayscale:
                stacked_obs = np.stack(list(self.frames), axis=-1)  # (H, W, stack)
            else:
                stacked_obs = np.concatenate(list(self.frames), axis=-1)

            reward_ = rewards[self.agent] - rewards[self.agent_other]
            bonus_reward = 0

            # print("Ball: ", self._get_ball_side(self.ball_coords, obs))
            if self._get_ball_side(self.ball_coords, obs)==self.agent: 
                if self.ball_touched:
                    bonus_reward += 0.08
                # elif not self.ball_touched:
                #     bonus_reward -= 0.001
            elif self._get_ball_side(self.ball_coords, obs)==self.agent_other: 
                if self.ball_touched:
                    bonus_reward -= 0.03
                elif not self.ball_touched:
                    bonus_reward += 0.02

            # ---------------- potential-based shaping ------------------------
            ball_x, court_w = self.ball_coords[1], obs.shape[1]
            potential_now   = ball_x / court_w                     # Φ(s′)
            delta_pot       = potential_now - self.prev_potential  # Φ(s′) − Φ(s)
            shaping_bonus   = self.shaping_lambda * delta_pot

            if gamestate == 1: 
                self.is_serving = False
            elif gamestate == 2:
                self.is_serving = True

            if self.is_serving: 
                bonus_reward += shaping_bonus  

            self.prev_potential = potential_now

            if self.step_count % self.anneal_every == 0:
                self.shaping_lambda *= self.anneal_factor
            # ---------------------------------------------------------------
            
            
            if gamestate == 1: # Left agent serve
                self.opponent_type = "attack"
            elif gamestate == 2: # Right agent serve
                self.opponent_type = "defend"
            elif gamestate == 3: # Left agent violated 5-second rule
                bonus_reward = 0
                reward_ = 0
            # elif gamestate == 4: # Right agent violated 5-second rule
            #     bonus_reward -=0.2
            elif gamestate == 5: # Left agent scored a point
                bonus_reward -=0.02

            done = False
            if (rewards[self.agent] + rewards[self.agent_other]) > 0:
                done = True
                print("Against opponent: ", self.opponent_policy)
                print("Gamestate: ", [gamestate])
                print ("Rewards: ", rewards[self.agent], rewards[self.agent_other], reward_ + bonus_reward)
                print("Action:", action)
                self._select_opponent()

            total_reward = np.clip(reward_ + bonus_reward, -2, 2)

            unmasked_obs = self._ori_preprocess(obs_dict[self.agent_obs]['observation'][0])
            infos[self.agent]["unmasked_obs"] = unmasked_obs

            return stacked_obs, total_reward, done, False, infos[self.agent]
        
        except Exception as e:
            print(f"Error in step: {e}")
            traceback.print_exc()
            raise

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
