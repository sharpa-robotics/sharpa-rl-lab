# environment setup
1. isaaclab:  
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html  
2. rl_isaaclab:  
```
cd sharpa_tac_rl 
pip install -e .
```

# step 1: generate grasp
```
python rl_isaaclab/scripts/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0 --headless
```
# step 2: train
```
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless
```
# step 3: distillation
```
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless --algorithm ProprioAdapt --load_path output
```

# visualization
## vis train
```
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 1 --load_path output
```
## vis distillation
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 1 --algorithm ProprioAdapt --load_path output