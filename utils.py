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