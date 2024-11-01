import random
import time
import uuid
import os
import json
import wandb
import wandb.sdk.data_types.video as wv
import numpy as np
import torch
from omegaconf import OmegaConf




def parse_cfg(cfg_path: str) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object."""
    base = OmegaConf.load(cfg_path)
    cli = OmegaConf.from_cli()
    for k,v in cli.items():
        if v == None:
            cli[k] = True
    base.merge_with(cli)
    return base


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Timer:
    def __init__(self):
        self.tik = None

    def start(self):
        self.tik = time.time()

    def stop(self):
        return time.time() - self.tik
    
    
class Logger:
    """Primary logger object. Logs in wandb."""
    def __init__(self, log_dir, cfg):
        self._log_dir = make_dir(log_dir)
        self._model_dir = make_dir(self._log_dir / 'models')
        self._video_dir = make_dir(self._log_dir / 'videos')
        self._cfg = cfg

        wandb.init(
            config=OmegaConf.to_container(cfg),
            project=cfg.project,
            group=cfg.group,
            name=cfg.exp_name,
            id=str(uuid.uuid4()),
            mode=cfg.wandb_mode,
            dir=self._log_dir
        )
        self._wandb = wandb

    def log(self, d, category):
        assert category in ['pretrain', 'train','inference']
        assert 'step' in d
        print(f"[{d['step']}]", " / ".join(f"{k} {v:f}" for k, v in d.items()))
        with (self._log_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": d['step'], **d}) + "\n")
        _d = {f"{category}/{k}": v for k, v in d.items()}
        self._wandb.log(_d, step=d['step'])
        
    def save_agent(self, agent=None, identifier='final'):
        if agent:
            fp = self._model_dir / f'model_{str(identifier)}.pt'
            agent.save(fp)
            print(f"model_{str(identifier)} saved")
    
    def save_diff_model(self, diff_model=None, identifier='final'):
        if diff_model:
            fp = self._model_dir / f'diff_model_{str(identifier)}.pt'
            diff_model.save(fp)
            print(f"diff_model_{str(identifier)} saved")
        
    def save_inv_model(self, inv_model=None, identifier='final'):
        if inv_model:
            fp = self._model_dir / f'inv_model_{str(identifier)}.pt'
            inv_model.save(fp)
            print(f"inv_model_{str(identifier)} saved")
    
    def finish(self):
        if self._wandb:
            self._wandb.finish()


    
    


    
