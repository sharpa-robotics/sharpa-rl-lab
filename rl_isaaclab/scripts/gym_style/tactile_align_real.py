import os
import sys
import time

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../utils/python'))
from sharpa import (
    SharpaWave,
    SharpaWaveManager,
    ControlMode,
    ControlSource,
    HandSide,
    ErrorCode
)

tac_uv_map = [np.load('assets/tactile_ha4_map/tactileSensor_map_4F_point.npy')] * 4
tac_uv_map.append(np.load('assets/tactile_ha4_map/tactileSensor_map_TH_point.npy'))

def auto_detect_hand() -> None:
    """Automatically detect device and return device and device serial number"""
    print("Searching for devices...")
    
    try:
        manager = SharpaWaveManager.get_instance()
        time.sleep(1)  # Wait for 1 seconds for device discovery to complete
        while True:
            devices = manager.get_all_device_sn()
            if not devices:
                print("No available devices found")
                time.sleep(1)
                continue
            else:
                print(f"Device found: {devices[0]}")
                return manager.connect(devices[0])
    except Exception as e:
        print(f"Failed to connect to device: {str(e)}")
        exit(1)

def initialize(hand) -> bool:
    error = hand.set_control_mode(ControlMode.POSITION)
    if error.code != 0:
        print(f"Failed to set control mode: {error.message}")
        return False
    error = hand.set_speed_coeff(0.5)
    if error.code != 0:
        print(f"Failed to set speed coeff: {error.message}")
        return False

    error = hand.set_current_coeff(0.4)
    if error.code != 0:
        print(f"Failed to set current coeff: {error.message}")
        return False
    error = hand.set_control_source(ControlSource.SDK)
    if error.code != 0:
        print(f"Failed to set control source: {error.message}")
        return False
    return True

def largest_connected_component_centroid(binary_image):
    """
    用 OpenCV 计算二维二值图像中面积最大的连通域的质心

    参数：
        binary_image (np.ndarray): 2D 二值图像，非零视为前景

    返回：
        centroid (tuple): (x, y) 的质心坐标
        mask (np.ndarray): 最大连通域的掩码
    """
    # 确保图像是 0/255 或 0/1
    img = (binary_image > 0).astype(np.uint8)

    # 计算连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

    if num_labels <= 1:
        return None, None  # 没有连通域

    # 忽略背景（标签0），找到面积最大的连通域
    # stats[:, cv2.CC_STAT_AREA] 返回每个连通域的像素面积
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1  # +1 因为排除了背景

    # 最大连通域掩码
    largest_mask = (labels == largest_idx).astype(np.uint8)

    # 最大连通域质心
    centroid = tuple(centroids[largest_idx])

    return centroid

def get_tactile_info(hand):
    force = [None] * 5
    contact_pos = [None] * 5
    while True:
        if None not in force: break
        for ch in range(5):
            ret = hand.fetch_tactile_frame(ch, timeout=0.1)
            if ret is None: continue
            deform_data = ret["content"].get("DEFORM")
            deform = deform_data.reshape(240, 240).astype(np.uint8)
            f6_data = ret["content"].get("F6")
            force[ch] = f6_data[:3]

            _, binary = cv2.threshold(deform, 30, 255, cv2.THRESH_BINARY)
            binary = binary.astype(np.uint8)
            # get largest connected component centroid
            center = largest_connected_component_centroid(binary)
            if center[0] is None or center[1] is None:
                contact_pos[ch] = np.array([np.nan, np.nan, np.nan])
            else:
                center_pos_ch = tac_uv_map[ch][int(center[0]), int(center[1])]
                contact_pos[ch] = np.array(center_pos_ch[:3]) / 1000.0
    force.reverse()
    contact_pos.reverse()
    print(torch.tensor(force).shape)
    print(torch.tensor(contact_pos).shape)
    tactile_info = torch.cat((torch.tensor(force), torch.tensor(contact_pos)), dim=-1)
    tactile_info = tactile_info.unsqueeze(0).unsqueeze(0)
    return tactile_info

def main():
    action_sequence_joint = [0, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 18, 20, 21]
    action_sequence = [0.0, 15.0, 30.0, 45.0, 60.0]
    force_collect = torch.zeros((1, 0, 5, 6), dtype=torch.float32)
    pos_diff = torch.zeros((1, 0, 22), dtype=torch.float32)

    hand = auto_detect_hand()
    if hand is None:
        print("Error: No available device found")
        exit(1)
    print("Sharpa Wave Example - Init Hand Running Mode")
    if not initialize(hand):
        print("Error: Failed to initialize hand")
        exit(1)
    hand.start()

    target_pos = np.zeros_like(hand.get_states().angles)
    for joint in action_sequence_joint:
        for pos in action_sequence:
            target_pos[:] = 0.0
            target_pos[joint] = np.deg2rad(pos)
            if joint == 0:
                target_pos[1] = np.deg2rad(pos)
            hand.set_joint_position(target_pos)
            time.sleep(1.0)
            cur_pos = hand.get_states().angles
            pos_diff = torch.cat((pos_diff, (torch.tensor(target_pos) - torch.tensor(cur_pos)).unsqueeze(0).unsqueeze(0)), dim=1)
            collect_data = get_tactile_info(hand)
            force_collect = torch.cat((force_collect, collect_data), dim=1)
            print("pos_diff: ", pos_diff.shape)
    print("collect done")
    np.save("cache/sharpa_tactile_align_tactile_info_real.npy", force_collect.numpy())
    np.save("cache/sharpa_tactile_align_pos_diff_real.npy", pos_diff.numpy())
    target_pos[:] = 0.0
    hand.set_joint_position(target_pos)
    exit()

if __name__ == "__main__":
    main()
