import os
import numpy as np
import torch
import sys
import glob
project_root = os.path.dirname(os.path.abspath("/home/zhouyi/repo/DA3-VGGT/"))
sys.path.insert(0, project_root)  

from depth_anything_3.api import DepthAnything3
from utils.viser_server import da3_prediction_to_viser_dict, viser_wrapper

def load_all_image(folder_path):
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png"), recursive=False) )
    # print(image_paths)
    image_paths.sort()
    return image_paths

def main():
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载DA3
    # model = DepthAnything3.from_pretrained("/path/to/DA3NESTED-GIANT-LARGE-1.1")
    model = DepthAnything3.from_pretrained("/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3-BASE")
    model = model.to(device)
    model.eval()
    
    # 准备输入图像
    folder_path ="/home/zhouyi/repo/dataset/sydney"
    image_paths = load_all_image(folder_path)
    
    
    # 推理
    print("Running DA3 inference...")
    prediction = model.inference(
        image=image_paths,
        process_res=504, 
        process_res_method="upper_bound_resize",
        export_dir=None,
    )
    
    # 检查输出形状
    print(f"Depth shape: {prediction.depth.shape}")
    print(f"Extrinsics shape: {prediction.extrinsics.shape}")
    print(f"Intrinsics shape: {prediction.intrinsics.shape}")
    print(f"Conf shape: {prediction.conf.shape}")
    
    # 转换数据格式
    print("Converting DA3 output to viser format...")
    pred_dict = da3_prediction_to_viser_dict(prediction, image_paths)
    
    # 可视化 使用从VGGT代码中提取的viser_wrapper
    print("Starting viser visualization...")
    viser_server = viser_wrapper(
        pred_dict=pred_dict,
        port=8080,                    # 网页服务器端口
        init_conf_threshold=60.0,     # conf的阈值 值越高就只输出高置信度的点云
        use_point_map=True,           # 使用计算出的world_points
        background_mode=False,        # 是否后台运行
        mask_sky=False,               # 是否进行天空分割 默认关闭
        image_folder=None             # 如果启用天空分割，需要提供图像文件夹路径  默认关闭
    )
    
    print("DA3 + Viser visualization is running!")

if __name__ == "__main__":
    main()

