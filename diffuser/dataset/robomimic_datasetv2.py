from pathlib import Path
import h5py
import numpy as np
import torch
import numba
from diffuser.dataset.dataset_utils import RotationTransformer
from diffuser.utils import MinMaxNormalizer

import zarr
import os
import shutil
import gc


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray, sequence_length: int, pad_before: int = 0, pad_after: int = 0, debug: bool = True
) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0  # episode start index
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]  # episode end index
        episode_length = end_idx - start_idx  # episode length

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

class SequenceSampler:
    def __init__(
        self,
        root_dir: str,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        zero_padding: bool = False,
        load_into_memory: bool = False,  # 新增参数
    ):
        super().__init__()
        assert sequence_length >= 1

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        self.root = zarr.open(root_dir, mode="r")
        self.load_into_memory = load_into_memory

        if keys is None:
            keys = list(self.root["data"].keys())
        
        episode_ends = self.root["meta"]["episode_ends"][:]

        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.indices = indices
        self.keys = list(keys)
        self.sequence_length = sequence_length
        self.zero_padding = zero_padding

        if self.load_into_memory:
            self.data_cache = {}
            for key in self.keys:
                self.data_cache[key] = self.root["data"][key][:]
            self.episode_ends = episode_ends
        else:
            self.data_cache = None

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = dict()
        for key in self.keys:
            if self.load_into_memory:
                input_arr = self.data_cache[key]
            else:
                input_arr = self.root["data"][key]
            sample = input_arr[buffer_start_idx:buffer_end_idx]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(shape=(self.sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
                if not self.zero_padding:
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < self.sequence_length:
                        data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result

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
        dataset_dir: str = "/home/yixin/Downloads/robomimic/datasets/lift/ph/image_abs.hdf5",
        shape_meta: str = {
            "robot0_eye_in_hand_image": "RGB",
            "agentview_image": "RGB",
            "robot0_eef_pos": "lowdim",
            "robot0_eef_quat": "lowdim",
            "robot0_gripper_qpos": "lowdim",
            "action": "action",
        },
        sequence_length=8,
        abs_action: bool = False,
    ):
        self.action_converter = ActionConverter(abs_action=abs_action)

        dataset, self.normalizers, self.info = self.get_dataset(dataset_dir, shape_meta)
        
        # save the dataset as zarr file
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        zarr_path = dataset_dir.replace(".hdf5", ".zarr")
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        zarr_dataset = zarr.open(zarr_path, mode="w", )
        data = zarr_dataset.create_group("data")
        meta = zarr_dataset.create_group("meta")

        for k, v in dataset["data"].items():
            if shape_meta[k] == "RGB":
                # (N, H, W, C) -> (N, C, H, W)
                data_ = dataset["data"][k].transpose(0, 3, 1, 2).astype(np.float32)/255.
            else:
                data_ = dataset["data"][k]
            data.create_dataset(
                name=k,
                data = data_,
                shape = (0,) + dataset["data"][k].shape[1:],
                chunks = (sequence_length,) + dataset["data"][k].shape[1:],
                dtype = np.float32,
                compressor = compressor,
            )
        meta.create_dataset(
            name = "episode_ends",
            data = dataset["meta"]["episode_ends"],
            shape = (0,),
            dtype = np.int64,
            compressor = compressor,
        )
        print('Successfully saved dataset to zarr file:', zarr_path)

        del dataset
        gc.collect()

        self.sampler = SequenceSampler(
            root_dir=zarr_path,
            sequence_length=sequence_length,
            pad_before=0,
            pad_after= sequence_length - 1,
            keys= shape_meta.keys(),
            zero_padding=False,
            load_into_memory=True,
        )

    def get_dataset(self, dataset_dir, meta):
        """
        Load dataset from hdf5 file based on meta,
        and normalize the lowdim and action data.
        """
        # 1. Load dataset from hdf5 file based on meta
        dataset = {"data": {"action": list()}, "meta": {"episode_ends": list()}}
        for k, v in meta.items():
            dataset["data"][k] = list()

        with h5py.File(dataset_dir, "r") as f:
            demos = f["data"]
            for i in range(len(demos)):
                demo = demos[f"demo_{i}"]

                for k, v in meta.items():
                    if "RGB" in v:
                        dataset["data"][k].append(demo["obs"][k][:].astype(np.uint8))
                    elif v == "lowdim":
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

        # 2. Normalize the lowdim and action data
        normailizers = {}
        for k, v in meta.items():
            if v == "lowdim":
                normailizers[k] = MinMaxNormalizer(dataset["data"][k])
                dataset["data"][k] = normailizers[k].normalize(dataset["data"][k])
        normailizers["action"] = MinMaxNormalizer(dataset["data"]["action"])

        # 3. Get info
        info = {}
        for k in dataset["data"].keys():
            info[k] = {
                "shape": dataset["data"][k].shape,
                "dtype": dataset["data"][k].dtype,
            }
        return dataset, normailizers, info
        
    def get_normalizers(self):
        return self.normalizers

    def undo_transform_action(self, action):
        return self.action_converter.inverse_transform(action)
    
    def __str__(self):
        return f"RobomimicOriginalDataset: {self.info}"

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        batch = self.sampler.sample_sequence(idx)
        return batch


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = RobomimicDataset(
        "/home/yixin/Downloads/robomimic/datasets/lift/ph/image.hdf5",
        shape_meta={
            "agentview_image": "RGB",
            "action": "action",
        },
        sequence_length=8,
        abs_action=False,
    )
    print(dataset)
    dataset = DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in dataset:
        import time
        t1 = time.time()
        act = batch["action"].to("cuda")
        image = batch["agentview_image"].to("cuda")
        print(time.time() - t1)
        