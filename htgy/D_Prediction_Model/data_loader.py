import numpy as np
import torch
from torch.utils.data import Dataset

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *

class SOCDataset(Dataset):
    def __init__(self, dynamic_path, static_path):
        dynamic_data = np.load(dynamic_path)
        static_data = np.load(static_path)
        
        # Dynamic variables (T, H, W)
        self.soc_fast = dynamic_data['soc_fast']
        self.soc_slow = dynamic_data['soc_slow']
        self.v_fast = dynamic_data['v_fast']
        self.v_slow = dynamic_data['v_slow']
        self.precip = dynamic_data['precip']
        self.check_dams = dynamic_data['check_dams']
        
        # Static variableds (H, W)
        self.dem = static_data['dem']
        self.loess_border_mask = static_data['loess_border_mask']
        self.river_mask = static_data['river_mask']
        self.small_boundary_mask = static_data['small_boundary_mask']
        self.large_boundary_mask = static_data['large_boundary_mask']
        self.small_outlet_mask = static_data['small_outlet_mask']
        self.large_outlet_mask = static_data['large_outlet_mask']
        
        self.T, self.H, self.W = self.soc_fast.shape
        
        assert self.T > 1, "Need at least 2 timesteps for backward prediction."
        
    def __len__(self):
        return self.T - 1

    def __getitem__(self, idx):
        # Input at time t
        x = np.stack([
            self.soc_fast[idx + 1],
            self.soc_slow[idx + 1],
            self.v_fast[idx + 1],
            self.v_slow[idx + 1],
            self.precip[idx + 1],
            self.check_dams[idx + 1],
        ], axis=0)  # shape: (6, H, W)
        
        # Static variables: expand to match input shape (add batch dimension)
        x = np.concatenate([
            x,
            self.dem[None, ...],         # shape: (1, H, W)
            self.loess_border_mask[None, ...],  # shape: (1, H, W)
            self.river_mask[None, ...],  # shape: (1, H, W)
            self.small_boundary_mask[None, ...],  # shape: (1, H, W)
            self.large_boundary_mask[None, ...],  # shape: (1, H, W)
            self.small_outlet_mask[None, ...],  # shape: (1, H, W)
            self.large_outlet_mask[None, ...],  # shape: (1, H, W)
        ], axis=0)  # Final shape: (8, H, W)
        
        # Ground truth: SOC at time t-1 (i.e., idx)
        y = np.stack([
            self.soc_fast[idx],
            self.soc_slow[idx],
        ], axis=0)  # shape: (2, H, W)
        
        x = torch.nan_to_num(torch.from_numpy(x).float(), nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(torch.from_numpy(y).float(), nan=0.0, posinf=0.0, neginf=0.0)
        
        return x, y