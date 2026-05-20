## 🚀 Overview
This is a repo for reinforcement learning sim2real rotation demo on SharpaWave, provides a step-by-step guide for training, visualizing and deploying.

<p align="center">
  <img src="resources/sim.gif" width="45%" />
  <img src="resources/real.gif" width="45%" />
</p>

## ⚠️ Environment Setup
### Follow the official  Isaaclab installation guide:
Install [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html). 

Ubuntu 22.04, conda environment, release/2.2.0 and release/2.3.0 have been tested.

CAUTION: A minimum of 32GB RAM is required. For specific requirements, please refer to [requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).
### Install this repo:  
```bash
conda activate env_isaaclab 
cd sharpa_tac_rl 
pip install -e .
```

## 📖 Training
### Generate grasp cache
```bash
# CAUTION: Same object scale config will overwrite the older one.
python rl_isaaclab/scripts/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0 --headless
```
### Train the policy
```bash
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless
```
### Distillation
```bash
# last.pth is recommended if curriculum is enabled
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless --algorithm ProprioAdapt --load_path ${pth}
```

## ▶️ Visualization
### Visualize trained policy
```bash
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --load_path ${pth}
```
### Visualize distillated policy
```bash
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --algorithm ProprioAdapt --load_path ${pth}
```

## 📝 Deploy
### Prepare SharpaWave and object
1. Calibrate SharpaWave through SharpaPilot. 
2. A cylinder with radius of 24mm and height of 60mm via 3D priting is recommended under default configuration.
### Deploy on SharpaWave (HostComputer Tactile, Recommended)
#### Configure docker
```bash
# INFOℹ️: Install docker and nvidia-ctk following steps 1-4 in <Steps to Acquire 180 Hz High-Frame-Rate High-Performance Tactile Information>.
cd rl_isaaclab/utils
# Configure docker-compose, substitute ${sharpa-rl-lab} with this repo path.
xhost +local:root
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose up -d
docker exec -it sharpawave_rl_dev bash
rm -r ~/sharpawave-rl-lab/rl_isaaclab/utils/python/
cp -r ~/sharpa-wave-sdk/python/sharpa/ ~/sharpawave-rl-lab/rl_isaaclab/utils/python/
cd ~/sharpawave-rl-lab/
python3 -m pip install -e .
```
#### Deploy
```bash
# INFOℹ️: Keyboard control is enabled by default. Press 'e' to start, press 'w' to freeze, press 'q' to go home.
python3 rl_isaaclab/scripts/deploy.py --task Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0 --hand_side ${0/1} --load_path ${pth}
```
### Deploy on SharpaWave (OnBoard Tactile)
#### Configure SharpaWaveSDK (For Deploy)
```bash
# INFOℹ️: Install SharpaWaveSDK following the official user manual. ${SharpaWaveSDK} is the root path of the SDK.
rm -r rl_isaaclab/utils/python
cp -r ${SharpaWaveSDK}/python rl_isaaclab/utils/python
```
####Deploy
```bash
# INFOℹ️: Keyboard control is enabled by default. Press 'e' to start, press 'w' to freeze, press 'q' to go home.
python rl_isaaclab/scripts/deploy.py --task Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0 --enable_on_board --hand_side ${0/1} --load_path ${pth}
```

## 🔄 Configure your own task via modifying the config file
Please refer to rl_isaaclab/tasks/inhand_rotate/sharpa_wave_env_cfg.py and rl_isaaclab/tasks/inhand_rotate/sharpa_wave_deploy_env_cfg.py for details.
