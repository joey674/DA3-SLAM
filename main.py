# main.py
import os
import sys
import argparse
import time
from config import load_config
from solver import SLAMSolver

def main():
    parser = argparse.ArgumentParser(description="DA3-SLAM: Monocular SLAM with Depth Anything 3")
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=False,
        default="/home/zhouyi/repo/dataset/sydney", 
        # default="/home/zhouyi/repo/dataset/statue", 
        help="Image path"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="/home/zhouyi/repo/DA3-SLAM/configs/config1.yaml",
        help="Image path",
    )

    args = parser.parse_args()
    
    config = load_config(args.config)
    image_dir = args.image_dir
    
    # 检查路径
    if not os.path.exists(image_dir):
        print(f"Error: Image folder {image_dir} does not exist!")
        sys.exit(1)
    
    # 导入路径设置
    project_root = os.path.dirname(os.path.abspath("/home/zhouyi/repo/DA3-SLAM/"))
    sys.path.insert(0, project_root)
    
    # 运行SLAM
    solver = SLAMSolver(image_dir,config)
    
    # 运行SLAM
    solver.run()
    
    
    print("SLAM running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("stopped by user")
    


if __name__ == "__main__":
    main()