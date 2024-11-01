import h5py
import numpy as np
import torch

from latent_diffuser.dataset.dataset_utils import ReplayBuffer, RotationTransformer, SequenceSampler, dict_apply

from latent_diffuser.utils import MinMaxNormalizer

from collections import defaultdict

class ActionConverter:
    def __init__(self, abs_action: bool = True, rotation_rep: str = "rotation_6d"):
        self.rotation_tf = RotationTransformer(from_rep="axis_angle", to_rep=rotation_rep)
        self.abs_action = abs_action

    def transform(self, action: np.ndarray):
        leading_dim = action.shape[:-1]
        if self.abs_action:
            is_dual_arm = action.shape[-1] == 14
            if is_dual_arm:
                action = action.reshape(*leading_dim, 2, 7)

            pos = action[..., :3]
            rot = action[..., 3:6]
            gripper = action[..., 6:]
            rot = self.rotation_tf.forward(rot)
            
            action = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

            if is_dual_arm:
                action = action.reshape(*leading_dim, 20)
        return action

    def inverse_transform(self, action: np.ndarray):
        leading_dim = action.shape[:-1]
        if self.abs_action:
            is_dual_arm = (action.shape[-1] == 20)
            if is_dual_arm:
                action = action.reshape(*leading_dim, 2, 10)

            d_rot = action.shape[-1] - 4
            pos = action[..., :3]
            rot = action[..., 3 : 3 + d_rot]
            gripper = action[..., [-1]]
            rot = self.rotation_tf.inverse(rot)
            action = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

            if is_dual_arm:
                action = action.reshape(*leading_dim, 14)
        return action


class RobomimicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str = "/home/yixin/Workspace/mimicgen/datasets/core/test.hdf5",
        shape_meta: str = {
            "action": {
                "shape": [7]
            },
            "obs": {
                "agentview_image": {
                    "shape": [3, 84, 84],
                    "type": "rgb",
                }
            }
            },
        horizon: int =16,
        abs_action: bool =False,
    ):
        # self.To, self.Ta = To, Ta
        self.action_converter = ActionConverter(abs_action=abs_action)

        # --- Get keys for rgb and lowdim data ---
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys


        # --- Load dataset ---
        dataset = {"data": {"action": list()}, "meta": {"episode_ends": list()}}
        
        for k, v in obs_shape_meta.items():
            dataset["data"][k] = list()
            dataset["data"][k] = list()
        

        # --- Load data from hdf5 file ---
        with h5py.File(dataset_dir, "r") as f:
            demos = f["data"]
            for i in range(len(demos)):
                demo = demos[f"demo_{i}"]

                for k, v in obs_shape_meta.items():
                    if v['type'] == 'rgb':
                        dataset["data"][k].append(demo["obs"][k][:].astype(np.uint8))
                    elif v['type'] == 'low_dim':
                        dataset["data"][k].append(demo["obs"][k][:].astype(np.float32))
                dataset["data"]["action"].append(demo["actions"][:].astype(np.float32))
                dataset["meta"]["episode_ends"].append(demo["actions"].shape[0])

        for k in dataset["data"].keys():
            if k == "action":
                dataset["data"]["action"] = self.action_converter.transform(
                    np.concatenate(dataset["data"]["action"], axis=0)
                )
            else:
                dataset["data"][k] = np.concatenate(dataset["data"][k], axis=0)
        dataset["meta"]["episode_ends"] = np.cumsum(dataset["meta"]["episode_ends"])

        # --- Create replay buffer and sampler ---
        self.replay_buffer = ReplayBuffer(dataset)
        self.sampler = SequenceSampler(
            self.replay_buffer,
            sequence_length= horizon,
            pad_before= 1,
            pad_after= 8 - 1,
            rgb_keys= self.rgb_keys,
        )


        # --- Create normalizers ---
        self.normalizer = self.get_normalizers()

        
        

    def get_normalizers(self):
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            normalizer['obs'][key] = MinMaxNormalizer(self.replay_buffer[key][:])

        normalizer['action'] = MinMaxNormalizer(self.replay_buffer['action'][:])

        return normalizer
        

    def __str__(self):
        return f"RobomimicDataset"

    def __len__(self):
        return len(self.sampler)

    
    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
    
        img_dict = {}
        for key in self.rgb_keys:
            img = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 255.0
            img_dict[key] = torch.from_numpy(img).contiguous()
            del sample[key]
        
        obs_dict = {}
        for key in self.lowdim_keys:
            obs = sample[key].astype(np.float32)
            obs = self.normalizer['obs'][key].normalize(obs)
            obs_dict[key] = torch.from_numpy(obs).contiguous()
            del sample[key]
    
        # action
        action = sample['action'].astype(np.float32)
        action = self.normalizer['action'].normalize(action)
        action = torch.from_numpy(action).contiguous()
    
        torch_data = {
            'img': img_dict,
            'obs': obs_dict,
            'action': action
        }
        return torch_data

    
    def undo_transform_action(self, action):
        action = self.action_converter.inverse_transform(action)
        return action

# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     shape_meta = {
#     "action": {
#         "shape": [7]
#     },
#     "obs": {
#         "robot0_joint_pos_cos": {
#             "shape": [7],
#             "type": "low_dim"
#         },
#         "robot0_joint_pos_sin": {
#             "shape": [7],
#             "type": "low_dim"
#         },
#         "object": {
#             "shape": [10],
#             "type": "low_dim"
#         },
#      }
#     }
#     dataset = RobomimicDataset(
#         dataset_dir="/home/yixin/Downloads/robomimic_low_dim/datasets/lift/mh/low_dim.hdf5",
#         shape_meta=shape_meta,
#         abs_action=False,
#     )


#     # sample a batch
#     dataloader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=True)
#     for batch in dataloader:
#         import time
#         t1 = time.time()
#         print(batch['obs']['robot0_joint_pos_cos'].shape)
#         print(batch['obs']['robot0_joint_pos_sin'].shape)
#         print(batch['obs']['object'].shape)
#         print(batch['action'].shape)
#         print(time.time() - t1)
#         break

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    shape_meta = {
    "action": {
        "shape": [7]
    },
    "obs": {
        "agentview_image": {
            "shape": [3, 84, 84],
            "type": "rgb",
        }
    }
    }
    dataset = RobomimicDataset(
        dataset_dir="/home/yixin/Downloads/robomimic/datasets/lift/mh/image.hdf5",
        shape_meta=shape_meta,
        abs_action=False,
        horizon=8,
    )

    # sample a batch
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=True)
    for batch in dataloader:
        import matplotlib.pyplot as plt

        # Convert images to byte format
        images = batch['img']['agentview_image'][0] * 255.0  # shape (T, C, H, W)
        images = images.byte()

        # Compute the frame differences
        diffs = images[1:] - images[:-1]

        # Plot all images and their corresponding differences
        num_images = images.shape[0]
        fig, axes = plt.subplots(2, num_images, figsize=(6 * (num_images - 1), 10 ))

        for t in range(num_images):
            axes[0, t].imshow(images[t].permute(1, 2, 0))
            # axes[t, 0].set_title(f"Image at t={t}")
            if t < num_images - 1:
                axes[1, t].imshow(diffs[t].permute(1, 2, 0))
            # axes[t, 1].set_title(f"Difference at t={t}")

        
        plt.show()

# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     shape_meta = {
#     "action": {
#         "shape": [7]
#     },
#     "obs": {
#         "agentview_image": {
#             "shape": [3, 84, 84],
#             "type": "rgb",
#         }
#     }
#     }
#     dataset = RobomimicDataset(
#         dataset_dir="/home/yixin/Downloads/robomimic/datasets/lift/mh/image.hdf5",
#         shape_meta=shape_meta,
#         abs_action=False,
#         horizon=8,
#     )

#     # sample a batch
#     dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=True)
#     for batch in dataloader:
#         import time
#         t1 = time.time()
#         images = batch['img']['agentview_image'].to('cuda:0')
#         print(time.time() - t1)