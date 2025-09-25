# environment setup
1. isaaclab:  
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html  
2. rl_isaaclab:  
```bash
conda activate env_isaaclab 
cd sharpa_tac_rl 
git submodule update --init --recursive 
pip install -e .
```

# step 1: generate grasp
```bash
python rl_isaaclab/scripts/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0 --headless
```
# step 2: train
```bash
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless
```
# step 3: distillation
```bash
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless --algorithm ProprioAdapt --load_path output
```

# visualization
## vis train
```bash
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --load_path output
```
## vis distillation
```bash
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --algorithm ProprioAdapt --load_path output
```

## deploy
```bash
python rl_isaaclab/scripts/deploy.py --task Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0 --load_path output
```

# make prim uninstanceable
```bash
python rl_isaaclab/scripts/make_uninstanceable.py --usd_file usd_file --prim_path prim_path
```