import numpy as np
import matplotlib.pyplot as plt

sim_tactile_info = np.load("cache/sharpa_tactile_align_tactile_info.npy")
sim_pos_diff = np.load("cache/sharpa_tactile_align_pos_diff.npy")
real_tactile_info = np.load("cache/sharpa_tactile_align_tactile_info_real.npy")
real_pos_diff = np.load("cache/sharpa_tactile_align_pos_diff_real.npy")

action_sequence_joint = {
    "right_thumb_CMC_FE": [4, 0],
    "right_thumb_MCP_FE": [14, 2],
    "right_thumb_IP": [21, 4],
    "right_index_MCP_FE": [0, 5],
    "right_index_PIP": [10, 7],
    "right_index_DIP": [15, 8],
    "right_middle_MCP_FE": [1, 9],
    "right_middle_PIP": [11, 11],
    "right_middle_DIP": [16, 12],
    "right_ring_MCP_FE": [3, 13],
    "right_ring_PIP": [13, 15],
    "right_ring_DIP": [18, 16],
    "right_pinky_MCP_FE": [7, 18],
    "right_pinky_PIP": [17, 20],
    "right_pinky_DIP": [20, 21],
}

action_sequence_joint_same_finger = {
    "right_thumb_CMC_FE": [[4, 14, 21], [0, 2, 4]],
    "right_thumb_MCP_FE": [[4, 14, 21], [0, 2, 4]],
    "right_thumb_IP": [[4, 14, 21], [0, 2, 4]],
    "right_index_MCP_FE": [[0, 10, 15], [5, 7, 8]],
    "right_index_PIP": [[0, 10, 15], [5, 7, 8]],
    "right_index_DIP": [[0, 10, 15], [5, 7, 8]],
    "right_middle_MCP_FE": [[1, 11, 16], [9, 11, 12]],
    "right_middle_PIP": [[1, 11, 16], [9, 11, 12]],
    "right_middle_DIP": [[1, 11, 16], [9, 11, 12]],
    "right_ring_MCP_FE": [[3, 13, 18], [13, 15, 16]],
    "right_ring_PIP": [[3, 13, 18], [13, 15, 16]],
    "right_ring_DIP": [[3, 13, 18], [13, 15, 16]],
    "right_pinky_MCP_FE": [[7, 17, 20], [18, 20, 21]],
    "right_pinky_PIP": [[7, 17, 20], [18, 20, 21]],
    "right_pinky_DIP": [[7, 17, 20], [18, 20, 21]],
}

joint_name2finger = {
    "right_thumb_CMC_FE": 0,
    "right_thumb_MCP_FE": 0,
    "right_thumb_IP": 0,
    "right_index_MCP_FE": 1,
    "right_index_PIP": 1,
    "right_index_DIP": 1,
    "right_middle_MCP_FE": 2,
    "right_middle_PIP": 2,
    "right_middle_DIP": 2,
    "right_ring_MCP_FE": 3,
    "right_ring_PIP": 3,
    "right_ring_DIP": 3,
    "right_pinky_MCP_FE": 4,
    "right_pinky_PIP": 4,
    "right_pinky_DIP": 4,
}

sim_pos_diff = np.mean(sim_pos_diff, axis=0)
sim_tactile_info = np.mean(sim_tactile_info, axis=0)
real_pos_diff = real_pos_diff[0]
# np.savetxt("cache/sharpa_tactile_align_pos_diff.txt", np.rad2deg(sim_pos_diff))
real_tactile_info = real_tactile_info[0]

fig_id = 0
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
# for joint_name, joint_id in action_sequence_joint.items():
#     ax = axes.flat[fig_id]
#     ax.set_title(f"{joint_name}")
#     ax.plot(sim_pos_diff[fig_id*5:fig_id*5+5, joint_id[0]], np.linalg.norm(sim_tactile_info[fig_id*5:fig_id*5+5, joint_name2finger[joint_name], :3], axis=-1), label='sim') 
#     ax.plot(real_pos_diff[fig_id*5:fig_id*5+5, joint_id[1]], np.linalg.norm(real_tactile_info[fig_id*5:fig_id*5+5, joint_name2finger[joint_name], :3], axis=-1), label='real')
#     ax.legend()
#     ax.set_xlabel('pos_diff')
#     ax.set_ylabel('f_norm')
#     fig_id += 1

for joint_name, joint_id in action_sequence_joint_same_finger.items():
    ax = axes.flat[fig_id]
    ax.set_title(f"{joint_name}")
    # ax.plot([0, 15, 30, 45, 60], np.rad2deg(real_pos_diff[fig_id*5:fig_id*5+5, joint_id[1][0]]), label='mcp_real')
    # ax.plot([0, 15, 30, 45, 60], np.rad2deg(real_pos_diff[fig_id*5:fig_id*5+5, joint_id[1][1]]), label='pip_real')
    # ax.plot([0, 15, 30, 45, 60], np.rad2deg(real_pos_diff[fig_id*5:fig_id*5+5, joint_id[1][2]]), label='dip_real')
    ax.plot([0, 15, 30, 45, 60], np.rad2deg(sim_pos_diff[fig_id*5:fig_id*5+5, joint_id[0][0]]), label='mcp_sim')
    ax.plot([0, 15, 30, 45, 60], np.rad2deg(sim_pos_diff[fig_id*5:fig_id*5+5, joint_id[0][1]]), label='pip_sim')
    ax.plot([0, 15, 30, 45, 60], np.rad2deg(sim_pos_diff[fig_id*5:fig_id*5+5, joint_id[0][2]]), label='dip_sim')
    ax.legend()
    ax.set_xlabel('tar_pos')
    ax.set_ylabel('pos_diff')
    fig_id += 1


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

