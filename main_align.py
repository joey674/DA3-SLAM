from depth_anything_3.api import DepthAnything3
import numpy as np
import time
import torch
from viewer import SLAMViewer
from align_geometry import (
    images_to_chw01,
    estimate_depth_scale,
    extract_overlap_point_cloud,
    align_two_point_clouds,
    compute_aligned_chunk_extrinsics_from_prev_overlap,
    make_image_chunks
)
from utils import (load_image,get_distinct_color,apply_chunk_color_to_images_batch)

def get_cur_chunk_global_extrinsics(overlap_frame_global_extrinsics_for_next_align,
                                     prev_chunk_prediction,
                                     cur_chunk_prediction,
                                     method: str = 'icp'):
    """
    Args:
        overlap_frame_global_extrinsics_for_next_align: (3, 4) 用于对齐的上一 chunk 的 overlap frame 全局外参
        prev_chunk_prediction: 上一 chunk 的预测结果
        cur_chunk_prediction: 当前 chunk 的预测结果
    Returns:
        cur_chunk_global_extrinsics: (N, 3, 4) 当前 chunk 每帧的全局外参 
        overlap_frame_global_extrinsics_for_next_align: (3, 4) 用于下一次对齐的当前 chunk 的 overlap frame(last frame) 全局外参
    """
    assert method in ("icp", "umeyama", "turboreg", "irls"), f"Error: Unknown method: {method}"

    if method in ['icp','turboreg']:
        # 0) 深度尺度对齐
        s_depth = estimate_depth_scale(prev_chunk_prediction, cur_chunk_prediction, conf_th=0.2)
        cur_chunk_prediction.depth = cur_chunk_prediction.depth * s_depth

    # 1) 提取 overlap 点云
    prev_overlap_point_cloud, cur_overlap_point_cloud = extract_overlap_point_cloud(
        prev_chunk_prediction, cur_chunk_prediction
    )
    prev_overlap_point_cloud = prev_overlap_point_cloud.reshape(-1, 3)  # (1, H, W, 3) -> (M, 3)
    cur_overlap_point_cloud  = cur_overlap_point_cloud.reshape(-1, 3)   # (1, H, W, 3) -> (M, 3)

    # 2) 对齐点云 获得点云变换
    s, R, t = align_two_point_clouds(cur_overlap_point_cloud, prev_overlap_point_cloud,method = method)

    ##############
    # TODO 这里到时候拓展成统一的SIM(3)转换形式; 加上scale(对于刚体变换来说 s=1不用变换,套用简单的深度乘以尺度即可)
    # 要改的有
    #   point_cloud_transform;
    #   compute_aligned_chunk_extrinsics_from_prev_overlap; 
    ##############

    point_cloud_transform = np.eye(4, dtype=np.float64)     # (4, 4)
    point_cloud_transform[:3, :3] = R                       # (3, 3)
    point_cloud_transform[:3, 3]  = t                       # (3,1)

    # 3) 根据点云变换 推当前 chunk 每帧的global_extrinsics (N, 3, 4)
    cur_chunk_local_extrinsics = np.asarray(cur_chunk_prediction.extrinsics, dtype=np.float64)

    cur_chunk_global_extrinsics = compute_aligned_chunk_extrinsics_from_prev_overlap(
        overlap_frame_global_extrinsics_for_next_align,  # (3, 4)
        cur_chunk_local_extrinsics,        # (N, 3, 4)
        point_cloud_transform,             # (4, 4)
    )

    # 4) 下一次对齐使用当前 chunk 的最后一帧全局 extrinsics
    overlap_frame_global_extrinsics_for_next_align = cur_chunk_global_extrinsics[-1]

    return cur_chunk_global_extrinsics, overlap_frame_global_extrinsics_for_next_align

folder_path = "/Users/guanzhouyi/repos/MA/DA3-SLAM/dataset/2077/scene1"
model_path = "/Users/guanzhouyi/repos/MA/Model_DepthAnythingV3/checkpoints/DA3-SMALL"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(model_path).to(device)

    viewer = SLAMViewer(port=8080)
    sleep_time = 3
    chunk_size = 4
    overlap = 1         

    image_paths = load_image(folder_path)
    chunks = make_image_chunks(image_paths, chunk_size=chunk_size, overlap=overlap)# [List(chunk_size),List(chunk_size),...]
    
    ########################################
    # ---- chunkA (第一个块) ----
    ########################################
    chunkA = model.inference(image=chunks[0], use_ray_pose=True)
    #
    img_chw = images_to_chw01(chunkA.processed_images)# 真实颜色
    img_chw = apply_chunk_color_to_images_batch(img_chw, 0)# 为当前chunk应用测试颜色
    #
    viewer.add_frame(
        img_chw[0], chunkA.depth[0], chunkA.conf[0], chunkA.extrinsics[0], chunkA.intrinsics[0]
    )
    viewer.add_frame(
        img_chw[-1], chunkA.depth[-1], chunkA.conf[-1], chunkA.extrinsics[-1], chunkA.intrinsics[-1]
    )

    overlap_frame_global_extrinsics_for_next_align = chunkA.extrinsics[-1]  # [R | t] (3*4) 初始：用于对齐的全局外参
    prev_chunk = chunkA

    time.sleep(sleep_time)

    ########################################
    # ---- chunkB (后续块循环复用) ----
    ########################################
    for idx in range(1, len(chunks)):
        chunkB = model.inference(image=chunks[idx], use_ray_pose=True)
        #
        img_chw = images_to_chw01(chunkB.processed_images)# 真实颜色
        img_chw = apply_chunk_color_to_images_batch(img_chw, idx)# 为当前chunk应用测试颜色
        # 对齐点云
        cur_chunk_global_extrinsics, overlap_frame_global_extrinsics_for_next_align = \
            get_cur_chunk_global_extrinsics(
                overlap_frame_global_extrinsics_for_next_align, 
                prev_chunk, 
                chunkB, 
                method='icp'
            )
        # 可视化
        viewer.add_frame(
            img_chw[0], chunkB.depth[0], chunkB.conf[0], cur_chunk_global_extrinsics[0], chunkB.intrinsics[0]
        )
        viewer.add_frame(
            img_chw[-1], chunkB.depth[-1], chunkB.conf[-1], cur_chunk_global_extrinsics[-1], chunkB.intrinsics[-1]
        )
        prev_chunk = chunkB
        time.sleep(sleep_time)

        
if __name__ == "__main__":
    main()
    print("SLAM running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("stopped by user")