# step 1: generate grasp
```python rl_isaaclab/scripts/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0 --num_envs 16384 --headless```
# step 2: train
```python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16384 --headless```
# step 3: distillation
```python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16384 --headless --algorithm=ProprioAdapt --load_path=/home/renrenyuan/sharpa_tac_rl/logs/gym_style/debug/2025-08-14_17-27-04/stage1_nn/best.pth```