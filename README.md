# Overview
This is a repo for reinforcement learning sim2real demo on SharpaWave, and provides a step-by-step guide for training, visualizing and deploying.
# Environment Setup
## Step 1. Follow the official  Isaaclab installation guide:  
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html  
## Step 2. Install this repo:  
```bash
conda activate env_isaaclab 
cd sharpa_tac_rl 
pip install -e .
```

# Training
## Step 1: Generate grasp cache
```bash
python rl_isaaclab/scripts/gym_style/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0 --headless
```
## Step 2: Train the policy
```bash
python rl_isaaclab/scripts/gym_style/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless
```
## Step 3: Distillation
```bash
python rl_isaaclab/scripts/gym_style/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless --algorithm ProprioAdapt --load_path output
```

# Visualization
## Visualize trained policy
```bash
python rl_isaaclab/scripts/gym_style/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --load_path {pth}
```
## Visualize distillated policy
```bash
python rl_isaaclab/scripts/gym_style/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --algorithm ProprioAdapt --load_path {pth}
```

# Deploy
## Deploy on SharpaWave
```bash
python rl_isaaclab/scripts/gym_style/deploy.py --task Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0 --load_path {pth}
```
