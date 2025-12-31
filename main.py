# main.py
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="DA3-SLAM: Monocular SLAM with Depth Anything 3 (Grouping Mechanism)")
    parser.add_argument("--image_folder", type=str,
                        default="/home/zhouyi/repo/dataset/sydney",
                        # default="/home/zhouyi/repo/dataset/office_loop",
                        help="Path to folder containing images")
    parser.add_argument("--group_size", type=int, 
                        default=40,
                        help="Number of frames per group (default: 30)")
    parser.add_argument("--overlap_size", type=int, 
                        default=2,
                        help="Number of overlapping frames between groups (default: 15)")
    parser.add_argument("--port", type=int, 
                        default=8080,
                        help="Port for visualization server (default: 8080)")
    parser.add_argument("--no_vis", action="store_true",
                        help="Disable visualization")
    parser.add_argument("--model_name", type=str,
                        default="DA3-BASE",
                        # default="DA3NESTED-GIANT-LARGE-1.1",
                        help="model name for da3")
    
    args = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(args.image_folder):
        print(f"Error: Image folder {args.image_folder} does not exist!")
        sys.exit(1)
    
    # 导入路径设置
    project_root = os.path.dirname(os.path.abspath("/home/zhouyi/repo/DA3-SLAM/"))
    sys.path.insert(0, project_root)
    
    # 运行SLAM
    from solver import SLAMSolver
    
    solver = SLAMSolver(
        viewer_port=args.port,
        chunk_size=args.group_size,
        overlap_size=args.overlap_size,
        model_name=args.model_name)
    
    try:
        # 运行SLAM
        solver.run_slam(args.image_folder)
        
        # 保持可视化运行
        if not args.no_vis:
            print("\nVisualization server is running...")
            print(f"Open browser at: http://localhost:{args.port}")
            solver.viewer.run(background=False)
            
    except KeyboardInterrupt:
        print("\nSLAM process interrupted by user")
    except Exception as e:
        print(f"Error during SLAM: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()