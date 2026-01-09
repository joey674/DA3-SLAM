from depth_anything_3.api import DepthAnything3
import numpy as np
import time
from typing import List
from viewer import SLAMViewer
from align_geometry import (
    images_to_chw01,
    estimate_depth_scale,
    extract_overlap_point_cloud,
    align_two_point_clouds,
    compute_aligned_chunk_extrinsics_from_prev_overlap
)
from utils import load_image

def get_cur_chunk_global_extrinsics(overlap_frame_global_extrinsics_for_next_align,
                                     prev_chunk_prediction,
                                     cur_chunk_prediction):
    """
    Args:
        overlap_frame_global_extrinsics_for_next_align: (3, 4) 用于对齐的上一 chunk 的 overlap frame 全局外参
        prev_chunk_prediction: 上一 chunk 的预测结果
        cur_chunk_prediction: 当前 chunk 的预测结果
    Returns:
        cur_chunk_global_extrinsics: (N, 3, 4) 当前 chunk 每帧的全局外参 
        overlap_frame_global_extrinsics_for_next_align: (3, 4) 用于下一次对齐的当前 chunk 的 overlap frame(last frame) 全局外参
    """

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
    _, R, t = align_two_point_clouds(cur_overlap_point_cloud, prev_overlap_point_cloud)

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


def make_image_chunks(image_paths: List[str], chunk_size: int, overlap: int = 1) -> List[List[str]]:
    """
        把图片路径按 chunk_size 分块，相邻 chunk 共享 overlap 张图（默认 overlap=1。
        例如 chunk_size=4, overlap=1:
        [0,1,2,3], [3,4,5,6], [6,7,8,9], ...
    """
    assert chunk_size >= 2, "chunk_size 至少要 2（否则没法做 overlap 对齐）"
    assert 0 <= overlap < chunk_size, "overlap 必须满足 0 <= overlap < chunk_size"

    n = len(image_paths)
    if n < chunk_size:
        return []

    step = chunk_size - overlap

    starts = list(range(0, n - chunk_size + 1, step))
    # 确保最后一张能被覆盖到（如果没刚好落在末尾，补一个最后起点）
    last_start = n - chunk_size
    if starts[-1] != last_start:
        starts.append(last_start)

    return [image_paths[s : s + chunk_size] for s in starts]


def main():
    model = DepthAnything3.from_pretrained(
        "/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3-LARGE-1.1"
    ).to("cuda")

    viewer = SLAMViewer(port=8080)
    sleep_time = 3
    chunk_size = 2
    overlap = 1         
    folder_path = "/home/zhouyi/repo/dataset/statue"

    image_paths = load_image(folder_path)
    chunks = make_image_chunks(image_paths, chunk_size=chunk_size, overlap=overlap)
    # -------------------------
    # ---- chunkA (第一个块) ----
    # -------------------------
    chunkA = model.inference(image=chunks[0], use_ray_pose=True)
    img_chw = images_to_chw01(chunkA.processed_images)

    viewer.add_frame(
        img_chw[0], chunkA.depth[0], chunkA.conf[0], chunkA.extrinsics[0], chunkA.intrinsics[0]
    )
    viewer.add_frame(
        img_chw[-1], chunkA.depth[-1], chunkA.conf[-1], chunkA.extrinsics[-1], chunkA.intrinsics[-1]
    )

    # 初始：用于对齐的全局外参
    overlap_frame_global_extrinsics_for_next_align = chunkA.extrinsics[-1]  # [R | t] (3*4)
    prev_chunk = chunkA

    time.sleep(sleep_time)

    # --------------------------------
    # ---- chunkB (后续块循环复用) ----
    # --------------------------------
    for idx in range(1, len(chunks)):
        chunkB = model.inference(image=chunks[idx], use_ray_pose=True)
        img_chw = images_to_chw01(chunkB.processed_images)
        # 对齐点云
        cur_chunk_global_extrinsics, overlap_frame_global_extrinsics_for_next_align = get_cur_chunk_global_extrinsics(
            overlap_frame_global_extrinsics_for_next_align, prev_chunk, chunkB
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