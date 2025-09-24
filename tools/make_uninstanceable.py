from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import isaacsim.core.utils.stage as stage_utils
import isaaclab.sim as sim_utils

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    
    # 添加参数
    parser.add_argument("--usd_file", type=str, required=True, help="path to the USD file")
    parser.add_argument("--prim_path", type=str, required=True, help="prim path")

    return parser.parse_args()
def main(args):
    usd_file = args.usd_file
    abs_path = os.path.abspath(os.path.join(os.getcwd(), usd_file))
    stage_utils.open_stage(abs_path)
    stage = stage_utils.get_current_stage()

    sim_utils.make_uninstanceable(args.prim_path)
    stage.Save()

    # close sim app
    simulation_app.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)