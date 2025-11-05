# Overview
This is a repo for reinforcement learning sim2real rotation demo on SharpaWave, provides a step-by-step guide for training, visualizing and deploying.

This repo is based on Isaaclab and hora.

# Environment Setup
## Step 1. Follow the official  Isaaclab installation guide:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html  
## Step 2. Install this repo:  
```bash
conda activate env_isaaclab 
cd sharpa_tac_rl 
pip install -e .
cp -r ${SharpaWaveSDK}/python rl_isaaclab/utils/python
```

# Training
## Step 1: Generate grasp cache
```bash
python rl_isaaclab/scripts/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0 --headless
```
## Step 2: Train the policy
```bash
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless
```
## Step 3: Distillation
```bash
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless --algorithm ProprioAdapt --load_path ${pth}
```

# Visualization
## Visualize trained policy
```bash
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --load_path ${pth}
```
## Visualize distillated policy
```bash
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --algorithm ProprioAdapt --load_path ${pth}
```

# Deploy
## Deploy on SharpaWave (OnBoard Tactile)
```bash
python rl_isaaclab/scripts/deploy.py --task Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0 --enable_on_board --load_path ${pth}
```
## Deploy on SharpaWave (HostComputer Tactile)
### Step 1. Start tactile zmq_pub in docker
```bash
docker exec -it tactile_dev bash
cd tactile_ws/tactile_sdk
python py/app/tactile_pub_zmq.py
```
### Step 2. Deploy
```bash
python rl_isaaclab/scripts/deploy.py --task Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0 --load_path ${pth}
```
