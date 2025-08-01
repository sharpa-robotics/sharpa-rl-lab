import torch
import numpy as np

def to_torch(x, dtype=torch.float32, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")

def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

def _obs_mujoco2hora(obses):
    trans_indices = np.array([
        5, 6, 7, 8,
        9, 10, 11, 12,
        17, 18, 19, 20, 21,
        13, 14, 15, 16,
        0, 1, 2, 3, 4
    ], dtype=np.int8)
    obses = obses[trans_indices]
    return obses

def _action_hora2mujoco(actions):
    trans_indices = np.array([
        17, 18, 19, 20, 21,
        0, 1, 2, 3,
        4, 5, 6, 7,
        13, 14, 15, 16,
        8, 9, 10, 11, 12
    ], dtype=np.int8)
    actions = actions[trans_indices]
    return actions
    return {
        'thumb_CMC_FE': actions[0],
        'thumb_CMC_AA': actions[1],
        'thumb_MCP_FE': actions[2],
        'thumb_MCP_AA': actions[3],
        'thumb_IP': actions[4],
        'index_MCP_FE': actions[5],
        'index_MCP_AA': actions[6],
        'index_PIP': actions[7],
        'index_DIP': actions[8],
        'middle_MCP_FE': actions[9],
        'middle_MCP_AA': actions[10],
        'middle_PIP': actions[11],
        'middle_DIP': actions[12],
        'ring_MCP_FE': actions[13],
        'ring_MCP_AA': actions[14],
        'ring_PIP': actions[15],
        'ring_DIP': actions[16],
        'pinky_CMC': actions[17],
        'pinky_MCP_FE': actions[18],
        'pinky_MCP_AA': actions[19],
        'pinky_PIP': actions[20],
        'pinky_DIP': actions[21],
    }

sharpa_dof_lower = torch.from_numpy(np.array([
    -0.1745, -0.3491,  0.0000, 0.0000,
    -0.1745, -0.3491,  0.0000, 0.0000, 
    0.0000, -0.1745, -0.3491,  0.0000,  0.0000,
    -0.1745, -0.3491,  0.0000, 0.0000,
    0.0000, -0.3491, -0.4363, -0.1745, -0.1745, 
]))

sharpa_dof_upper = torch.from_numpy(np.array([
    1.5708, 0.3491, 1.5708, 1.5708,
    1.5708, 0.3491, 1.5708, 1.5708, 
    0.2618, 1.5708, 0.3491, 1.5708, 1.5708, 
    1.5708, 0.3491, 1.5708, 1.5708, 
    1.9199, 0.3491, 1.3090, 0.1745, 1.5708
]))