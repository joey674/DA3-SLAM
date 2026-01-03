import os
import glob
from typing import List

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