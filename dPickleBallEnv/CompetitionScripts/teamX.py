import numpy as np
from collections import deque
import cv2

import torch
from einops import rearrange

from STORM.utils import seed_np_torch, Logger, load_config
from STORM.replay_buffer import ReplayBuffer
import STORM.env_wrapper
import STORM.agents
from STORM.sub_models.functions_losses import symexp
from STORM.sub_models.world_models import WorldModel, MSELoss
# Build a Python class for your solution, do preprocessing (image processing, frame stacking, etc) here.
# During the competition, only the policy function is called at each time step, providing the observation and reward for that time step only.
# Your agent is expected to return actions to be executed.
class TeamX:
    # Define square directions
    square_directions = [
            np.array([1, 0, 0], dtype=np.int32),   # right
            np.array([0, 1, 0], dtype=np.int32),   # up
            np.array([2, 0, 0], dtype=np.int32),  # left
            np.array([0, 2, 0], dtype=np.int32)   # down
        ]
    steps_per_side = 30  # Number of env steps per side of the square
    step = 0

    def __init__(self, frame_stack=1):
        self.frames = deque(maxlen=frame_stack)

        conf = load_config("/home/marl/space/renice/STORM/config_files/unity_dp.yaml")
        from STORM import train
        action_dim = 9
        world_model = train.build_world_model(conf, action_dim)
        agent = train.build_agent(conf, action_dim)
        root_path = f"/home/marl/space/renice/STORM/ckpt/unity_dp_experiment"

        step = 10000
        self.world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
        self.agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))

        self.world_model.eval()
        self.agent.eval()
        self.context_obs = deque(maxlen=16)
        self.context_action = deque(maxlen=16)
        # Load your checkpoint for policy network
        # model.load()
    
    # Your policy takes only visual representation as input, 
    # and reward is 1 when you score, -1 when your opponent scores
    # Your policy function returns actions
    def policy(self, observation, reward):
        # Implement your solution here

        # image processing
        obs = observation.transpose(1, 2, 0)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
        self.frames.append(observation)

        stacked_obs = np.stack(list(self.frames), axis=-1)  # (H, W, stack)

        # Use your policy network here
        # model.predict()
        with torch.no_grad():
            if len(self.context_action) == 0:
                action = np.random.randint(0, 2, size=(1))  # Random action
            else:
                context_latent = self.world_model.encode_obs(torch.cat(list(self.context_obs), dim=1))
                model_context_action = np.stack(list(self.context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).cuda()
                prior_flattened_sample, last_dist_feat = self.world_model.calc_last_dist_feat(context_latent, model_context_action)
                action = self.agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                    greedy=False
                )

        self.context_obs.append(rearrange(torch.Tensor(obs).cuda(), "B H W C -> B 1 C H W")/255)
        self.context_action.append(action)

        # Convert flat index to multidiscrete
        dims = dims = np.array([3, 3, 3], dtype=np.int32)
        md_action = []
        for dim in reversed(dims):
            md_action.append(action % dim)
            action = action // dim
            
        return np.array(md_action[::-1])



