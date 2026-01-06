from depth_anything_3.api import DepthAnything3
from utils import (load_image)
from geometry import (depth_to_point_cloud_vectorized)
import numpy as np
import time

# Initialize and run inference
model = DepthAnything3.from_pretrained("/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3-LARGE-1.1").to("cuda")

from viewer import SLAMViewer
viewer = SLAMViewer(port=8080)

# image_paths = load_image("/home/zhouyi/repo/dataset/sydney")
image0 = "/home/zhouyi/repo/dataset/sydney/000000.png"
image1 = "/home/zhouyi/repo/dataset/sydney/000010.png"
image2 = "/home/zhouyi/repo/dataset/sydney/000020.png"
image2 = "/home/zhouyi/repo/dataset/sydney/000030.png"
image2 = "/home/zhouyi/repo/dataset/sydney/000040.png"


# 生成点云1
prev_chunk_prediction = model.inference(image=[image0, image1], use_ray_pose=True)
# print(f"{prev_chunk_prediction.depth.shape}")
# 提取原始图像 需要转换为CHW格式(不然颜色是反的)
image_hwc1 = prev_chunk_prediction.processed_images[0]  # [H, W, 3]
image_chw1 = image_hwc1.transpose(2, 0, 1) / 255.0
image_hwc2 = prev_chunk_prediction.processed_images[1]  # [H, W, 3]
image_chw2 = image_hwc2.transpose(2, 0, 1) / 255.0
viewer.add_frame(image_chw1,
                 prev_chunk_prediction.depth[0],
                 prev_chunk_prediction.conf[0],
                 prev_chunk_prediction.extrinsics[0],
                 prev_chunk_prediction.intrinsics[0],
                 1)
viewer.add_frame(image_chw2,
                 prev_chunk_prediction.depth[1],
                 prev_chunk_prediction.conf[1],
                 prev_chunk_prediction.extrinsics[1],
                 prev_chunk_prediction.intrinsics[1],
                 2)


time.sleep(3)

# 生成点云2
cur_chunk_prediction = model.inference(image=[image1, image2], use_ray_pose=True)
image_hwc1 = cur_chunk_prediction.processed_images[0]  # [H, W, 3]
image_chw1 = image_hwc1.transpose(2, 0, 1) / 255.0
image_hwc2 = cur_chunk_prediction.processed_images[1]  # [H, W, 3]
image_chw2 = image_hwc2.transpose(2, 0, 1) / 255.0
# 重叠点云提取
from test_align import (extract_overlap_point_cloud,
                        align_two_point_clouds,
                        get_transformed_extrisinc,
                        )
overlapped_pointcloud1, overlapped_pointcloud2 = extract_overlap_point_cloud(prev_chunk_prediction,cur_chunk_prediction) # [N(frames), H(height), W(width), 3]
overlapped_pointcloud1 = overlapped_pointcloud1.reshape(-1, 3)  #  [N, H, W, 3] => [N*H*W, 3]
overlapped_pointcloud2 = overlapped_pointcloud2.reshape(-1, 3)
# 计算点云变换矩阵
s,R,t = align_two_point_clouds(overlapped_pointcloud2,overlapped_pointcloud1) # 返回的是2=>1的变换  当前实现是So3 不是Sim3
print(f"s: {s}")
print(f"R: {R}")
print(f"t: {t}")
# 根据点云的变换矩阵 计算外参变换矩阵
transformed_extrinsic = get_transformed_extrisinc(prev_chunk_prediction.extrinsics[-1],s,R,t)
def to4x4(E3x4):
    E = np.eye(4, dtype=np.float64)
    E[:3, :4] = E3x4
    return E
def to3x4(E4x4):
    return E4x4[:3, :4]
E0_local = to4x4(cur_chunk_prediction.extrinsics[0])  # w2c (local)
E1_local = to4x4(cur_chunk_prediction.extrinsics[1])  # w2c (local)
E0_aligned = to4x4(transformed_extrinsic)             # w2c (aligned/world)
T_c1_c0 = E1_local @ np.linalg.inv(E0_local)
E1_aligned = T_c1_c0 @ E0_aligned
transformed_extrinsic0 = to3x4(E0_aligned)
transformed_extrinsic1 = to3x4(E1_aligned)
# visualize
viewer.add_frame(image_chw1,
                 cur_chunk_prediction.depth[0],
                 cur_chunk_prediction.conf[0],
                 transformed_extrinsic0,
                 cur_chunk_prediction.intrinsics[0],
                 3)
viewer.add_frame(image_chw2,
                 cur_chunk_prediction.depth[1],
                 cur_chunk_prediction.conf[1],
                 transformed_extrinsic1,
                 cur_chunk_prediction.intrinsics[1],
                 4)


# 运行
print("SLAM running. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(0.01)
except KeyboardInterrupt:
    print("stopped by user")
