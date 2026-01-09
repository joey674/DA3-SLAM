from depth_anything_3.api import DepthAnything3
import numpy as np
import time
from viewer import SLAMViewer
from align_geometry import (
    images_to_chw01,
    estimate_depth_scale,
    extract_overlap_point_cloud,
    align_two_point_clouds,
    compute_aligned_chunk_extrinsics_from_prev_overlap
)


def get_cur_chunk_global_extrinsics(overlap_frame_global_extrinsics_for_next_align,
                                     prev_chunk_prediction,
                                     cur_chunk_prediction):
    """
    用 overlap 帧把当前 chunk 对齐到全局坐标系

    -------------------------
    输入 (Inputs):
      overlap_frame_global_extrinsics_for_next_align: np.ndarray, shape = (3, 4)
        - 上一个 chunk overlap(最后一帧) 的“全局 w2c 外参”
        - w2c: world -> camera

      prev_chunk_prediction:
        - 上一 chunk 的预测对象
        - 至少包含:
            prev_chunk_prediction.depth:      (N_prev, H, W)
            prev_chunk_prediction.intrinsics: (N_prev, 3, 3)
            prev_chunk_prediction.extrinsics: (N_prev, 3, 4)  (w2c, local/global视你定义)
            prev_chunk_prediction.conf:       (N_prev, H, W)  (可选)

      cur_chunk_prediction:
        - 当前 chunk 的预测对象
        - 至少包含:
            cur_chunk_prediction.depth:      (N_cur, H, W)
            cur_chunk_prediction.intrinsics: (N_cur, 3, 3)
            cur_chunk_prediction.extrinsics: (N_cur, 3, 4)   (w2c, “局部”坐标系)
            cur_chunk_prediction.conf:       (N_cur, H, W)   (可选)

    -------------------------
    输出 (Returns):
      cur_chunk_global_extrinsics: np.ndarray, shape = (N_cur, 3, 4)
        - 当前 chunk 每一帧在“全局坐标系”下的 w2c 外参

      prev_overlap_frame_extrinsics_for_next: np.ndarray, shape = (3, 4)
        - 下一次对齐要用的 overlap 帧全局外参
        - 这里取 cur_chunk_global_extrinsics 的最后一帧：cur_chunk_global_extrinsics[-1]
    """

    # 0) 深度尺度对齐（只改当前chunk）
    # s_depth: float
    # cur_chunk_prediction.depth: (N_cur, H, W)
    s_depth = estimate_depth_scale(prev_chunk_prediction, cur_chunk_prediction, conf_th=0.2)
    cur_chunk_prediction.depth = cur_chunk_prediction.depth * s_depth

    # 1) overlap 点云（camera coords）
    # prev_overlap_point_cloud: (1, H, W, 3)  点在 prev_overlap 相机坐标系
    # cur_overlap_point_cloud : (1, H, W, 3)  点在 cur_overlap 相机坐标系
    prev_overlap_point_cloud, cur_overlap_point_cloud = extract_overlap_point_cloud(
        prev_chunk_prediction, cur_chunk_prediction
    )

    # reshape: (M, 3), M = H*W（或有效点数量）
    prev_overlap_point_cloud = prev_overlap_point_cloud.reshape(-1, 3)  # (M, 3)
    cur_overlap_point_cloud  = cur_overlap_point_cloud.reshape(-1, 3)   # (M, 3)

    # 2) 对齐点云 获得点云变换
    _, R, t = align_two_point_clouds(cur_overlap_point_cloud, prev_overlap_point_cloud)

    point_cloud_transform = np.eye(4, dtype=np.float64)     # (4, 4)
    point_cloud_transform[:3, :3] = R                       # (3, 3)
    point_cloud_transform[:3, 3]  = t                       # (3,)

    # 3) 根据点云变换 推当前 chunk 每帧的global_extrinsics (N_cur, 3, 4)
    cur_chunk_local_extrinsics = np.asarray(cur_chunk_prediction.extrinsics, dtype=np.float64)

    cur_chunk_global_extrinsics = compute_aligned_chunk_extrinsics_from_prev_overlap(
        overlap_frame_global_extrinsics_for_next_align,  # (3, 4)
        cur_chunk_local_extrinsics,        # (N_cur, 3, 4)
        point_cloud_transform,             # (4, 4)
    )

    # 4) 下一次对齐使用当前 chunk 的最后一帧全局 w2c
    # prev_overlap_frame_extrinsics_for_next: (3, 4)
    overlap_frame_global_extrinsics_for_next_align = cur_chunk_global_extrinsics[-1]

    return cur_chunk_global_extrinsics, overlap_frame_global_extrinsics_for_next_align



def main():
    model = DepthAnything3.from_pretrained(
        "/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3-LARGE-1.1"
        # "/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3NESTED-GIANT-LARGE-1.1"
    ).to("cuda")

    viewer = SLAMViewer(port=8080)
    sleep_time = 3

    # image0 = "/home/zhouyi/repo/dataset/sydney/000000.png"
    # image1 = "/home/zhouyi/repo/dataset/sydney/000010.png"
    # image2 = "/home/zhouyi/repo/dataset/sydney/000020.png"
    # image3 = "/home/zhouyi/repo/dataset/sydney/000030.png"
    image0 = "/home/zhouyi/repo/dataset/statue/frame_00007.png"
    image1 = "/home/zhouyi/repo/dataset/statue/frame_00068.png"
    image2 = "/home/zhouyi/repo/dataset/statue/frame_00113.png"
    image3 = "/home/zhouyi/repo/dataset/statue/frame_00205.png"
    # image0 = "/home/zhouyi/repo/dataset/C3VD2/brightness/cropped_00002201.jpg"
    # image1 = "/home/zhouyi/repo/dataset/C3VD2/brightness/cropped_00002220.jpg"
    # image2 = "/home/zhouyi/repo/dataset/C3VD2/brightness/cropped_00002240.jpg"
    # image3 = "/home/zhouyi/repo/dataset/C3VD2/brightness/cropped_00002260.jpg"


    # ---- chunkA [0,1]：定义初始坐标系并渲染 ----
    chunkA = model.inference(image=[image0, image1], use_ray_pose=True)

    img_chw = images_to_chw01(chunkA.processed_images)

    viewer.add_frame(img_chw[0], chunkA.depth[0], chunkA.conf[0], chunkA.extrinsics[0], chunkA.intrinsics[0])
    viewer.add_frame(img_chw[1], chunkA.depth[1], chunkA.conf[1], chunkA.extrinsics[1], chunkA.intrinsics[1])

    # 初始：用于对齐的全局外参
    overlap_frame_global_extrinsics_for_next_align = chunkA.extrinsics[-1] #  [R | t] (3*4)
    prev_chunk = chunkA

    time.sleep(sleep_time)


    # ---- chunkB [1,2]：对齐到初始坐标系并渲染 ----
    chunkB = model.inference(image=[image1, image2], use_ray_pose=True)

    img_chw = images_to_chw01(chunkB.processed_images)

    # 对齐点云
    cur_chunk_global_extrinsics, overlap_frame_global_extrinsics_for_next_align = get_cur_chunk_global_extrinsics(
        overlap_frame_global_extrinsics_for_next_align, prev_chunk, chunkB
    )

    viewer.add_frame(img_chw[0], chunkB.depth[0], chunkB.conf[0], cur_chunk_global_extrinsics[0], chunkB.intrinsics[0])
    viewer.add_frame(img_chw[1], chunkB.depth[1], chunkB.conf[1], cur_chunk_global_extrinsics[1], chunkB.intrinsics[1])

    prev_chunk = chunkB

    time.sleep(sleep_time)

    # # ---- chunkC [2,3]：对齐到初始坐标系并渲染 ----
    # chunkC = model.inference(image=[image2, image3], use_ray_pose=True)

    # img2c = images_to_chw01(chunkC, 0)
    # img3c = images_to_chw01(chunkC, 1)

    # # 先做尺度对齐
    # s_depth = estimate_depth_scale(prev_chunk, chunkC, conf_th=0.2)
    # chunkC.depth = chunkC.depth * s_depth  

    # # 对齐点云
    # E0_c, E1_c, prev_overlap_aligned,s = get_aligned_two_frame_extrinsics(
    #     prev_overlap_aligned, prev_chunk, chunkC
    # )

    # viewer.add_frame(img2c, chunkC.depth[0], chunkC.conf[0], E0_c, chunkC.intrinsics[0])
    # viewer.add_frame(img3c, chunkC.depth[1], chunkC.conf[1], E1_c, chunkC.intrinsics[1])

    # prev_chunk = chunkC


        
if __name__ == "__main__":
    main()
    print("SLAM running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("stopped by user")