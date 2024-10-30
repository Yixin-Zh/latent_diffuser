from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.
# import ctypes
# ctypes.CDLL("libX11.so").XInitThreads()

import hydra
import os
# os.environ['PYOPENGL_PLATFORM']  = 'osmesa'
# os.environ['HYDRA_FULL_ERROR'] = '1'
import sys
import warnings
warnings.filterwarnings('ignore')

import gym
import pathlib
import time
import collections
import numpy as np
import torch
import torch.nn as nn
from diffuser.util import set_seed, parse_cfg
from torch.optim.lr_scheduler import CosineAnnealingLR


from diffuser.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from diffuser.env.async_vector_env import AsyncVectorEnv
from diffuser.env.utils import VideoRecorder
from diffuser.dataset.robomimic_datasetv2 import RobomimicDataset
from diffuser.dataset.dataset_utils import loop_dataloader
from diffuser.utils import report_parameters


from diffuser.bc.bc_img_agengv2 import Agent



def make_async_envs(args):
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    
    print(f"Starting to create {args.num_envs} asynchronous Robomimic environments...")

    def create_robomimic_env(env_meta, shape_meta, enable_render=True):
        modality_mapping = collections.defaultdict(list)
        for key, attr in shape_meta['obs'].items():
            modality_mapping[attr.get('type', 'low_dim')].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=True, 
            render_offscreen=enable_render,
            use_image_obs=enable_render, 
        )
        return env
    
    dataset_path = os.path.expanduser(args.datapath)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    # disable object state observation
    env_meta['env_kwargs']['use_object_obs'] = False
    abs_action = args.abs_action  
    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    def env_fn():
        env = create_robomimic_env(
            env_meta=env_meta, 
            shape_meta=args.shape_meta
        )
        # Robosuite's hard reset causes excessive memory consumption.
        # Disabled to run more envs.
        # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
        env.env.hard_reset = False
        return MultiStepWrapper(
            VideoRecordingWrapper(
                RobomimicImageWrapper(
                    env=env,
                    shape_meta=args.shape_meta,
                    init_state=None,
                    render_obs_key=args.render_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=10,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=22,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=2
            ),
            n_obs_steps=args.obs_steps,
            n_action_steps=args.action_steps,
            max_episode_steps=args.max_episode_steps
        )
    
    # See https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env_runner/robomimic_image_runner.py
    # For each process the OpenGL context can only be initialized once
    # Since AsyncVectorEnv uses fork to create worker process,
    # a separate env_fn that does not create OpenGL context (enable_render=False)
    # is needed to initialize spaces.
    def dummy_env_fn():
        env = create_robomimic_env(
                env_meta=env_meta, 
                shape_meta=args.shape_meta,
                enable_render=True
            )
        return MultiStepWrapper(
            VideoRecordingWrapper(
                RobomimicImageWrapper(
                    env=env,
                    shape_meta=args.shape_meta,
                    init_state=None,
                    render_obs_key=args.render_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=10,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=22,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=2
            ),
            n_obs_steps=args.obs_steps,
            n_action_steps=args.action_steps,
            max_episode_steps=args.max_episode_steps
        )
    
    env_fns = [env_fn] * args.num_envs
    # env_fn() and dummy_env_fn() should be function!
    envs = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
    envs.seed(args.seed)
   
    return envs 
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
def create_robomimic_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=True, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env

@hydra.main(config_path=".", config_name="bc.yaml", )
def pipeline(args):

    # ---------------- Create Environment ----------------
    dataset_path = os.path.expanduser(args.datapath)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    # disable object state observation
    env_meta['env_kwargs']['use_object_obs'] = True
    abs_action = args.abs_action  
    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    env = create_robomimic_env(
            env_meta=env_meta, 
            shape_meta=args.robomimic_meta
        )
    # ---------------- Create Dataset ----------------
    datapath = args.datapath
    dataset =RobomimicDataset(dataset_dir=datapath, shape_meta= args.shape_meta,
                                sequence_length=8,
                                abs_action=args.abs_action,)
    
    # ---------------- Create Agent ----------------
   
    agent = Agent(latent_dim=256, action_dim=7, device=args.device)
    agent.load(os.path.join(args.model_path, 'model_latest'))
    agent.eval()


    # ---------------- Create Diffuser ----------------
   
    # ---------------- Start Rollout ----------------
    episode_rewards = []
    episode_steps = []
    episode_success = []
    
    video = []
    gen_video = []
    for i in range(3): 
        ep_reward = 0.0
        obs, t = env.reset(), 0

        
        while t < 200:
            # normalize observation
            img = obs['agentview_image']
            
         
            actions, gen_img = agent.sample_action(img, deterministic=True)
            
            # Denormalize prediction
            action_pred = dataset.normalizers['action'].unnormalize(actions)
            action = action_pred
            print(action)
            if args.abs_action:
                action = dataset.undo_transform_action(action)
            obs, reward, done, info = env.step(action)
            video_img = env.render(mode="rgb_array", height=512, width=512, camera_name="agentview")
        
            video.append(video_img)
            gen_video.append(gen_img)
            print(f"t: {t} reward: {reward}")
            ep_reward += reward
            t += 1
            if done:
                break
            

        
        success = 1.0 if ep_reward > 0 else 0.0
        print(f"[Episode {i+1}] reward: {np.around(ep_reward, 2)} success: {success}")
        episode_rewards.append(ep_reward)
        episode_steps.append(t)
        episode_success.append(success)

    print(f"Mean step: {np.nanmean(episode_steps)} Mean reward: {np.nanmean(episode_rewards)} Mean success: {np.nanmean(episode_success)}")

    
    video_filename = 'episode_video.mp4'  
    fps = 30  
    import imageio
   
    imageio.mimsave(video_filename, video, fps=fps)
    gen_video_filename = 'gen_video.mp4'
    imageio.mimsave(gen_video_filename, gen_video, fps=fps)

   


if __name__ == "__main__":
    pipeline()


