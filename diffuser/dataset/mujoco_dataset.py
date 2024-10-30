from typing import Dict

import numpy as np
import torch
import os

from diffuser.dataset.base_dataset import BaseDataset
from diffuser.utils import MinMaxNormalizer, dict_apply
from torch.utils.data import DataLoader


class D4RLMuJoCoDataset(BaseDataset):
    """ **D4RL-MuJoCo Sequential Dataset**

    torch.utils.data.Dataset wrapper for D4RL-MuJoCo dataset.
    Chunk the dataset into sequences of length `horizon` without padding.
    Use GaussianNormalizer to normalize the observations as default.
    Each batch contains
    - batch["obs"]["state"], observations of shape (batch_size, horizon, o_dim)
    - batch["act"], actions of shape (batch_size, horizon, a_dim)
    - batch["rew"], rewards of shape (batch_size, horizon, 1)
    - batch["val"], Monte Carlo return of shape (batch_size, 1)

    Args:
        dataset: Dict[str, np.ndarray],
            D4RL-MuJoCo dataset. Obtained by calling `env.get_dataset()`.
        terminal_penalty: float,
            Penalty reward for early-terminal states. Default is -100.
        horizon: int,
            Length of each sequence. Default is 1.
        max_path_length: int,
            Maximum length of the episodes. Default is 1000.
        discount: float,
            Discount factor. Default is 0.99.

    Examples:
        >>> env = gym.make("halfcheetah-medium-expert-v2")
        >>> dataset = D4RLMuJoCoDataset(env.get_dataset(), horizon=32)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> batch = next(iter(dataloader))
        >>> obs = batch["obs"]["state"]  # (32, 32, 17)
        >>> act = batch["act"]           # (32, 32, 6)
        >>> normalizer = dataset.get_normalizer()
        >>> obs = env.reset()[None, :]
        >>> normed_obs = normalizer.normalize(obs)
        >>> unnormed_obs = normalizer.unnormalize(normed_obs)
    """
    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            terminal_penalty: float = -100.,
            horizon: int = 1,
            max_path_length: int = 1000,
    ):
        super().__init__()

        observations, actions, timeouts, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["timeouts"],
            dataset["terminals"])
        self.normalizers = {
            "state": MinMaxNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        n_paths = np.sum(np.logical_or(terminals, timeouts))
        self.seq_obs = np.zeros((n_paths, max_path_length, self.o_dim), dtype=np.float32)
        self.seq_act = np.zeros((n_paths, max_path_length, self.a_dim), dtype=np.float32)
        self.indices = []

        path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i]:
                path_lengths.append(i - ptr + 1)

                self.seq_obs[path_idx, :i - ptr + 1] = normed_observations[ptr:i + 1]
                self.seq_act[path_idx, :i - ptr + 1] = actions[ptr:i + 1]

                max_start = min(path_lengths[-1] - 1, max_path_length - horizon)
                self.indices += [(path_idx, start, start + horizon) for start in range(max_start + 1)]

                ptr = i + 1
                path_idx += 1

        self.path_lengths = np.array(path_lengths)

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        only take state-action pairs
        """
        path_idx, start, end = self.indices[idx]

        data = {
            'obs': {
                'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data
    

class VD4RLDatasetBaseDataset(BaseDataset):
    """ **D4RL-MuJoCo Sequential Dataset**

    torch.utils.data.Dataset wrapper for D4RL-MuJoCo dataset.
    Chunk the dataset into sequences of length `horizon` without padding.
    Use GaussianNormalizer to normalize the observations as default.
    Each batch contains
    """
    def __init__(
            self,
            datapath: str,
            horizon: int = 1,
            max_path_length: int = 1000,
    ):
        super().__init__()
        
        dataset = self.get_dataset(datapath)
        observations, actions, timeouts, terminals = (
            dataset["observations"].astype(np.float32)/255.,
            dataset["actions"].astype(np.float32),
            dataset["timeouts"],
            dataset["terminals"])

        self.horizon = horizon
        self.o_dim = observations.shape[1:]
        self.a_dim = actions.shape[-1]

        n_paths = np.sum(np.logical_or(terminals, timeouts))
        self.seq_obs = np.zeros((n_paths, max_path_length)+ self.o_dim, dtype=np.float32)
        self.seq_act = np.zeros((n_paths, max_path_length, self.a_dim), dtype=np.float32)
        self.indices = []

        path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i]:
                path_lengths.append(i - ptr + 1)

                self.seq_obs[path_idx, :i - ptr + 1] = observations[ptr:i + 1]
                self.seq_act[path_idx, :i - ptr + 1] = actions[ptr:i + 1]

                max_start = min(path_lengths[-1] - 1, max_path_length - horizon)
                self.indices += [(path_idx, start, start + horizon) for start in range(max_start + 1)]

                ptr = i + 1
                path_idx += 1

        self.path_lengths = np.array(path_lengths)

    def get_dataset(self, dir):
        print("Loading dataset from", dir)
        dataset = {"observations": [], "actions": [], "timeouts": [], "terminals": []}
        for filename in os.listdir(dir):
            if filename.endswith('.npz'):
                file_path = os.path.join(dir, filename)
                data = np.load(file_path)
                dataset["observations"].append(data["image"])
                dataset["actions"].append(data["action"])
                dataset["timeouts"].append(data["is_last"])
                dataset["terminals"].append(data["is_terminal"])
        dataset["observations"] = np.concatenate(dataset["observations"], axis= 0)
        dataset["actions"] = np.concatenate(dataset["actions"], axis= 0)
        dataset["timeouts"] = np.concatenate(dataset["timeouts"], axis= 0)
        dataset["terminals"] = np.concatenate(dataset["terminals"], axis= 0)
        return dataset


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        only take state-action pairs
        """
        path_idx, start, end = self.indices[idx]

        data = {
            'obs': self.seq_obs[path_idx, start:end].transpose(0, 3, 1, 2),
            'act': self.seq_act[path_idx, start:end],
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data
    
# if __name__ == '__main__':
#     dir = '/home/yixin/Downloads/vd4rl/main/walker_walk/expert/64px/'
#     dataset = VD4RLDatasetBaseDataset(dir, horizon=16)
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#     batch = next(iter(dataloader))

#     print("Observations shape:", batch["obs"].shape)  # Expected: (32, 32, o_dim)
#     print("Actions shape:", batch["act"].shape)  # Expected: (32, 32, a_dim)
