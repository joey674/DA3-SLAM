from depth_anything_3.api import DepthAnything3
import numpy as np
import torch
import time
from typing import List
from viewer import SLAMViewer
from align_geometry import (
    make_image_chunks,
    images_to_chw01,
)
from utils import load_image
import matplotlib.pyplot as plt
from PIL import Image

###
folder_path = "/Users/guanzhouyi/repos/MA/DA3-SLAM/dataset/2077/scene1"
model_path = "/Users/guanzhouyi/repos/MA/Model_DepthAnythingV3/checkpoints/DA3-LARGE"
output_path = "/Users/guanzhouyi/repos/MA/DA3-SLAM/output/"


def print_conf_stats(conf_data: np.ndarray):
    """
    输出置信度的统计信息（不打印详细信息）
    """
    conf_array = np.array(conf_data)
    
    if conf_array.ndim > 1:
        conf_flat = conf_array.flatten()
    else:
        conf_flat = conf_array
    
    # 根据实际数据范围动态生成区间
    data_min, data_max = np.min(conf_flat), np.max(conf_flat)
    bin_width = (data_max - data_min) / 5
    bins = [data_min + i * bin_width for i in range(6)]
    
    return bins, data_min, data_max

def create_confidence_comparison(img_data, conf_data, bins, save_path="/tmp/confidence_comparison.png"):
    """
    创建置信度对比图 - 只显示原图、heatmap和去掉最低两个区间后的图像
    """
    # 转换图像数据格式
    if isinstance(img_data, torch.Tensor):
        img_np = img_data.detach().cpu().numpy()
    else:
        img_np = np.array(img_data)
    
    # 确保图像是HWC格式
    if img_np.shape[0] == 3:  # CHW格式
        img_np = img_np.transpose(1, 2, 0)
    
    # 归一化图像到0-1范围
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    
    # 转换置信度数据
    conf_np = np.array(conf_data)
    
    # 创建子图 - 只显示3个
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Confidence-based Image Comparison', fontsize=16, fontweight='bold')
    
    # 原始图像
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 置信度热力图
    im1 = axes[1].imshow(conf_np, cmap='viridis', vmin=bins[0], vmax=bins[-1])
    axes[1].set_title('Confidence Heatmap')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 筛选掉最低的两个区间（保留置信度较高的区域）
    high_conf_mask = conf_np > bins[2]  # 去掉最低的2个区间，只保留后3个区间
    img_high_conf = img_np.copy()
    img_high_conf[~high_conf_mask] = 0  # 低置信度区域变黑
    total_high_conf_pixels = np.sum(high_conf_mask)  # 高置信度区域的像素数
    axes[2].imshow(img_high_conf)
    axes[2].set_title(f'High Confidence Region (> {bins[2]:.3f})\nVisible pixels: {total_high_conf_pixels:,}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_overall_heatmap(all_conf_data, save_path="/tmp/overall_heatmap.png"):
    """
    创建所有帧的heatmap排列图（n帧就n个heatmap排列显示）
    """
    # 转换数据格式
    if isinstance(all_conf_data, list):
        all_conf = all_conf_data  # 保持为列表
    else:
        all_conf = [all_conf_data[i] for i in range(all_conf_data.shape[0])]
    
    n_frames = len(all_conf)
    
    # 计算子图布局
    cols = min(4, n_frames)  # 最多4列
    rows = (n_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'All {n_frames} Frames Confidence Heatmaps', fontsize=16, fontweight='bold')
    
    # 为每帧创建heatmap
    for i in range(n_frames):
        conf_np = all_conf[i]
        min_val = np.min(conf_np)
        max_val = np.max(conf_np)
        
        im = axes[i].imshow(conf_np, cmap='viridis', vmin=min_val, vmax=max_val)
        axes[i].set_title(f'Frame {i}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for i in range(n_frames, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"所有帧热力图排列已保存到: {save_path}")

def main():
    device = "cpu"
    model = DepthAnything3.from_pretrained(model_path).to(device)

    chunk_size = 9
    image_paths = load_image(folder_path)
    chunks = make_image_chunks(image_paths, chunk_size=chunk_size)

    chunkA = model.inference(image=chunks[0], use_ray_pose=True)
    img_chw = images_to_chw01(chunkA.processed_images)


    for i in range(len(chunkA.conf)):
        bins_i, min_i, max_i = print_conf_stats(chunkA.conf[i])
        create_confidence_comparison(
            img_chw[i], 
            chunkA.conf[i], 
            bins_i, 
            f"{output_path}frame{i}_confidence_comparison.png"
        )
    create_overall_heatmap(chunkA.conf, output_path+"overall_heatmaps.png")


if __name__ == "__main__":
    main()
    print("CONF running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("stopped by user")