import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os

from STORM.utils import seed_np_torch, Logger, load_config
from STORM.replay_buffer import ReplayBuffer
import STORM.env_wrapper as env_wrapper
import STORM.agents as agents
from STORM.sub_models.functions_losses import symexp
from STORM.sub_models.world_models import WorldModel, MSELoss

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel

from STORM.mylib import SharedObsUnityGymWrapper, CustomCNN, FlattenActionWrapper, LimitToMoveOnlyActionWrapper

SAVE_VIDEO_EVERY_STEPS = 200

def build_single_env(env_name, image_size, seed):
    string_channel = StringSideChannel()
    channel = CustomDataChannel()
    #init points
    reward_cum = [0,0]
    # Decode game config code
    # The last digit is the serve mode:
    #     1 = left serve only
    #     2 = right serve only
    #     3 = random serve
    # The remaining digits (before the last one) indicate the match point.
    # Example: 213 → match point = 21, serve = 3 (random serve)
    # Example: 12 → match point = 1, serve = 2 (right serve)
    channel.send_data(serve=12, p1=reward_cum[0], p2=reward_cum[1])
    print("Hello dPickleBall Trainer")

    unity_env = UnityEnvironment("/home/marl/space/renice/build_linux_V2/dp.x86_64", side_channels=[string_channel, channel])

    print("environment created")
    env = SharedObsUnityGymWrapper(unity_env)
    env = LimitToMoveOnlyActionWrapper(env)
    return env


def build_vec_env(env_name, image_size, num_envs, seed):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size, seed)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, demonstration_batch_size, batch_length, logger):
    obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
    world_model.update(obs, action, reward, termination, logger=logger)


@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, agent: agents.ActorCriticAgent,
                             imagine_batch_size, imagine_demonstration_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()

    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    return latent, action, None, None, reward_hat, termination_hat


def joint_train_world_model_agent(env_name, max_steps, num_envs, image_size,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel, agent: agents.ActorCriticAgent,
                                  train_dynamics_every_steps, train_agent_every_steps,
                                  batch_size, demonstration_batch_size, batch_length,
                                  imagine_batch_size, imagine_demonstration_batch_size,
                                  imagine_context_length, imagine_batch_length,
                                  save_every_steps, seed, logger):
    # create ckpt dir
    os.makedirs(f"ckpt/{args.n}", exist_ok=True)

    import cv2
    import datetime
    video_writer = None
    record_video = False
    frame_size = (168, 84)
    fps = 30

    recording_video = False
    video_frame_count = 0
    video_max_frames = 200


    # build vec env, not useful in the Atari100k setting
    # but when the max_steps is large, you can use parallel envs to speed up
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, seed=seed)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)

    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    try:
        # sample and train
        for total_steps in tqdm(range(max_steps//num_envs)):
            # sample part >>>
            if replay_buffer.ready():
                world_model.eval()
                agent.eval()
                with torch.no_grad():
                    if len(context_action) == 0:
                        action = vec_env.action_space.sample()
                    else:
                        context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                        model_context_action = np.stack(list(context_action), axis=1)
                        model_context_action = torch.Tensor(model_context_action).cuda()
                        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                        action = agent.sample_as_env_action(
                            torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                            greedy=False
                        )

                context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
                context_action.append(action)
            else:
                action = vec_env.action_space.sample()

            obs, reward, done, truncated, info = vec_env.step(action)
            if done or truncated:
                print(f"Current reward: {reward}, done: {done}, truncated: {truncated}")
            replay_buffer.append(current_obs, action, reward, done)

            if record_video:
                img = current_obs[0, ..., -1]  # (84, 168), last grayscale frame in stack
                img = img - img.min()
                if img.max() > 0:
                    img = img / img.max()
                else:
                    img = np.zeros_like(img)

                img = (img * 255).astype(np.uint8)  # Normalize to 0–255
                img = np.stack([img] * 3, axis=-1)  # Grayscale to RGB
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video_writer.write(img_bgr)

            done_flag = np.logical_or(done, truncated)
            if done_flag.any():
                for i in range(num_envs):
                    if done_flag[i]:
                        logger.log(f"sample/{env_name}_reward", sum_reward[i])
                        logger.log(f"sample/{env_name}_episode_steps", current_info["episode_frame_number"][i]//4)  # framskip=4
                        logger.log("replay_buffer/length", len(replay_buffer))
                        sum_reward[i] = 0

            # update current_obs, current_info and sum_reward
            sum_reward += reward
            current_obs = obs
            current_info = info
            # <<< sample part

            # train world model part >>>
            if replay_buffer.ready() and total_steps % (train_dynamics_every_steps//num_envs) == 0:
                train_world_model_step(
                    replay_buffer=replay_buffer,
                    world_model=world_model,
                    batch_size=batch_size,
                    demonstration_batch_size=demonstration_batch_size,
                    batch_length=batch_length,
                    logger=logger
                )
            # <<< train world model part

            # train agent part >>>
            if replay_buffer.ready() and total_steps % (train_agent_every_steps//num_envs) == 0 and total_steps*num_envs >= 0:
                if total_steps % (save_every_steps//num_envs) == 0:
                    log_video = True
                else:
                    log_video = False

                if total_steps % (SAVE_VIDEO_EVERY_STEPS // num_envs) == 0 and not recording_video:
                    # Start recording
                    record_video = True
                    recording_video = True
                    video_frame_count = 0
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = f"videos/video_step{total_steps}_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
                elif recording_video:
                    record_video = True
                    video_frame_count += 1
                    if video_frame_count >= video_max_frames:
                        # Stop recording
                        record_video = False
                        recording_video = False
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                else:
                    record_video = False



                imagine_latent, agent_action, agent_logprob, agent_value, imagine_reward, imagine_termination = world_model_imagine_data(
                    replay_buffer=replay_buffer,
                    world_model=world_model,
                    agent=agent,
                    imagine_batch_size=imagine_batch_size,
                    imagine_demonstration_batch_size=imagine_demonstration_batch_size,
                    imagine_context_length=imagine_context_length,
                    imagine_batch_length=imagine_batch_length,
                    log_video=log_video,
                    logger=logger
                )

                agent.update(
                    latent=imagine_latent,
                    action=agent_action,
                    old_logprob=agent_logprob,
                    old_value=agent_value,
                    reward=imagine_reward,
                    termination=imagine_termination,
                    logger=logger
                )
            # <<< train agent part

            # save model per episode
            if total_steps % (save_every_steps//num_envs) == 0:
                print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
                torch.save(world_model.state_dict(), f"ckpt/{args.n}/world_model_{total_steps}.pth")
                torch.save(agent.state_dict(), f"ckpt/{args.n}/agent_{total_steps}.pth")

    finally:
        vec_env.close()
        if video_writer:
            video_writer.release()


def build_world_model(conf, action_dim):
    return WorldModel(
        in_channels=conf.Models.WorldModel.InChannels,
        action_dim=action_dim,
        transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
        transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
        transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
        transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads
    ).cuda()


def build_agent(conf, action_dim):
    return agents.ActorCriticAgent(
        feat_dim=32*32+conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
    ).cuda()


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from xvfbwrapper import Xvfb
    # Start virtual display
    vdisplay = Xvfb(width=1024, height=768)
    vdisplay.start()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-trajectory_path", type=str, required=True)
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # set seed
    seed_np_torch(seed=args.seed)
    # tensorboard writer
    logger = Logger(path=f"runs/{args.n}")
    # copy config file
    shutil.copy(args.config_path, f"runs/{args.n}/config.yaml")

    # distinguish between tasks, other debugging options are removed for simplicity
    if conf.Task == "JointTrainAgent":
        # getting action_dim with dummy env
        dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize, seed=0)
        action_dim = dummy_env.action_space.n
        frame_stack = dummy_env.env.frame_stack
        print("Action dimension: ", action_dim)
        dummy_env.close()

        # build world model and agent
        world_model = build_world_model(conf, action_dim)
        agent = build_agent(conf, action_dim)

        # build replay buffer
        replay_buffer = ReplayBuffer(
            obs_shape=(conf.BasicSettings.ImageSize[0], conf.BasicSettings.ImageSize[1], frame_stack),
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_length=conf.JointTrainAgent.BufferMaxLength,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU
        )

        # judge whether to load demonstration trajectory
        if conf.JointTrainAgent.UseDemonstration:
            print(colorama.Fore.MAGENTA + f"loading demonstration trajectory from {args.trajectory_path}" + colorama.Style.RESET_ALL)
            replay_buffer.load_trajectory(path=args.trajectory_path)

        # train
        try:
            joint_train_world_model_agent(
                env_name=args.env_name,
                num_envs=conf.JointTrainAgent.NumEnvs,
                max_steps=conf.JointTrainAgent.SampleMaxSteps,
                image_size=conf.BasicSettings.ImageSize,
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                train_dynamics_every_steps=conf.JointTrainAgent.TrainDynamicsEverySteps,
                train_agent_every_steps=conf.JointTrainAgent.TrainAgentEverySteps,
                batch_size=conf.JointTrainAgent.BatchSize,
                demonstration_batch_size=conf.JointTrainAgent.DemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
                batch_length=conf.JointTrainAgent.BatchLength,
                imagine_batch_size=conf.JointTrainAgent.ImagineBatchSize,
                imagine_demonstration_batch_size=conf.JointTrainAgent.ImagineDemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
                imagine_context_length=conf.JointTrainAgent.ImagineContextLength,
                imagine_batch_length=conf.JointTrainAgent.ImagineBatchLength,
                save_every_steps=conf.JointTrainAgent.SaveEverySteps,
                seed=args.seed,
                logger=logger
            )
        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            vdisplay.stop()
    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")
