import os
import glob
from typing import List
import torch
import numpy as np

def load_image(folder_path: str) -> List[str]:
    """加载图像路径列表"""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext), recursive=False))
    
    # 排序 (按数字顺序 )
    def extract_number(filename):
        base = os.path.basename(filename)
        name, _ = os.path.splitext(base)
        numbers = ''.join(filter(str.isdigit, name))
        return int(numbers) if numbers else 0
    
    image_paths.sort(key=extract_number)
    
    if not image_paths:
        print(f"Warning: No images found in {folder_path}")
        return []
    
    print(f"Found {len(image_paths)} images in {folder_path}")
    return image_paths


def extract_keyframe(image_paths: List[str], num_keyframe: int) -> List[str]:
    """
    从图像路径列表中按照固定间隔抽取关键帧
    
    Args:
        image_paths: 排序后的图像路径列表
        num_keyframe: 抽取间隔，每隔num_keyframe张抽取一张
    
    Returns:
        抽取后的关键帧路径列表
    """
    if not image_paths:
        return []
    
    if num_keyframe <= 0:
        print(f"Warning: num_keyframe must be positive, got {num_keyframe}")
        return image_paths  # 如果间隔为0或负数，返回所有帧
    
    # 使用切片操作，从第一个开始每隔num_keyframe抽取一张
    keyframe_paths = image_paths[::num_keyframe]
    
    print(f"Extracted {len(keyframe_paths)} keyframes from {len(image_paths)} total frames "
          f"(interval={num_keyframe})")
    
    return keyframe_paths



# for test
# 全局颜色变量
chunk_colors = []
def get_distinct_color(chunk_idx):
    """
    为每个chunk生成不同的亮色，相邻chunk颜色对比大
    """
    # 预设8个高亮对比色（红、绿、蓝、黄、洋红、青、橙、紫）
    bright_colors = [
        (1.0, 0.0, 0.0),  # 红
        (0.0, 1.0, 0.0),  # 绿  
        (0.0, 0.0, 1.0),  # 蓝
        (1.0, 1.0, 0.0),  # 黄
        (1.0, 0.0, 1.0),  # 洋红
        (0.0, 1.0, 1.0),  # 青
        (1.0, 0.5, 0.0),  # 橙
        (0.5, 0.0, 1.0),  # 紫
    ]
    
    return bright_colors[chunk_idx % len(bright_colors)]


def apply_chunk_color_to_images_batch(img_chw, chunk_idx):
    """
    批量处理整个img_chw，给每个chunk的所有图像应用一个纯色
    """
    global chunk_colors
    
    # 如果这个chunk还没有颜色，生成一个
    if chunk_idx >= len(chunk_colors):
        color = get_distinct_color(chunk_idx)
        chunk_colors.append(color)
    else:
        color = chunk_colors[chunk_idx]
    
    # 转换图像格式
    if isinstance(img_chw, torch.Tensor):
        img_np = img_chw.detach().cpu().numpy()
    else:
        img_np = np.array(img_chw)
    
    # 确保图像是HWC格式
    if img_np.ndim == 4 and img_np.shape[1] == 3:  # (batch, 3, H, W) CHW格式
        img_np = img_np.transpose(0, 2, 3, 1)  # 转为 (batch, H, W, 3)
    
    # 归一化到0-1范围
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    
    # 应用纯色：所有像素都变成同一个颜色
    colored_imgs = []
    for i in range(img_np.shape[0]):
        pure_colored_img = np.ones_like(img_np[i]) * np.array(color).reshape(1, 1, 3)
        colored_imgs.append(pure_colored_img)
    
    print(f"Chunk {chunk_idx} apply color: RGB{tuple(int(c*255) for c in color)}")
    return colored_imgs