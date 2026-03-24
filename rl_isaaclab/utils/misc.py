import threading

import torch
import numpy as np

def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


class AverageScalarMeter(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0).cpu().numpy().item()
        size = np.clip(size, 0, self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean


class ThreadSafeValue:
    def __init__(self, value=None):
        self._value = value
        self._lock = threading.Lock()

    def set(self, value):
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


_ISAACLAB2SHARPA_IDX = torch.tensor(
    [4, 9, 14, 19, 21, 0, 5, 10, 15, 1, 6, 11, 16, 3, 8, 13, 18, 2, 7, 12, 17, 20],
    dtype=torch.long
)

_SHARPA2ISAACLAB_IDX = torch.tensor(
    [5, 9, 17, 13, 0, 6, 10, 18, 14, 1, 7, 11, 19, 15, 2, 8, 12, 20, 16, 3, 21, 4],
    dtype=torch.long
)

def dof_isaaclab2sharpa(dof_pos: torch.Tensor) -> torch.Tensor:
    """
    dof_pos: (..., D)
    """
    idx = _ISAACLAB2SHARPA_IDX.to(dof_pos.device)
    return dof_pos.index_select(dim=-1, index=idx)

def dof_sharpa2isaaclab(dof_pos: torch.Tensor) -> torch.Tensor:
    """
    dof_pos: (..., D)
    """
    idx = _SHARPA2ISAACLAB_IDX.to(dof_pos.device)
    return dof_pos.index_select(dim=-1, index=idx)
