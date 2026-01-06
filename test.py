from depth_anything_3.api import DepthAnything3
import numpy as np
import time

from viewer import SLAMViewer

# 你自己的对齐工具（确保 extract_overlap_point_cloud 用 camera coords）
from test_align import (
    extract_overlap_point_cloud,   # prev最后一帧 vs cur第一帧，返回 camera coords 点云
    align_two_point_clouds,        # ICP: source->target, 返回 R,t
)

def to4x4(E3x4):
    E = np.eye(4, dtype=np.float64)
    E[:3, :4] = E3x4
    return E

def to3x4(E4x4):
    return E4x4[:3, :4]

def image_to_chw01(pred, idx):
    img_hwc = pred.processed_images[idx]          # [H,W,3] uint8
    return img_hwc.transpose(2, 0, 1) / 255.0     # [3,H,W] float

def get_aligned_two_frame_extrinsics(prev_overlap_aligned_3x4,
                                     prev_chunk_prediction,
                                     cur_chunk_prediction):
    """
    用 overlap 帧做 ICP，把当前 chunk 对齐到全局(初始)坐标系。
    prev_overlap_aligned_3x4: 上一个chunk overlap帧（最后一帧）的 w2c（已经是全局坐标系）
    返回：
      cur_frame0_aligned_3x4, cur_frame1_aligned_3x4, cur_overlap_aligned_lastframe_for_next
    """

    # 1) 取 overlap 点云（必须 camera coords）
    pc_prev, pc_cur = extract_overlap_point_cloud(prev_chunk_prediction, cur_chunk_prediction)
    pc_prev = pc_prev.reshape(-1, 3)
    pc_cur  = pc_cur.reshape(-1, 3)

    # 2) ICP：cur -> prev （2=>1）
    s, R, t = align_two_point_clouds(pc_cur, pc_prev)  # source=cur, target=prev
    T_prev_cur = np.eye(4, dtype=np.float64)
    T_prev_cur[:3, :3] = R
    T_prev_cur[:3, 3]  = t

    # 3) 由 prev 的全局外参推 cur overlap(第0帧) 的全局外参（w2c）
    E_prev_aligned = to4x4(prev_overlap_aligned_3x4)   # w2c(prev overlap, global)
    E0_aligned = np.linalg.inv(T_prev_cur) @ E_prev_aligned  # w2c(cur frame0, global)

    # 4) 用 chunk 内相对位姿把 frame1 推出来（仍然是 w2c）
    E0_local = to4x4(cur_chunk_prediction.extrinsics[0])  # w2c(local)
    E1_local = to4x4(cur_chunk_prediction.extrinsics[1])  # w2c(local)

    T_c1_c0 = E1_local @ np.linalg.inv(E0_local)          # c0 -> c1 (w2c形式的相对)
    E1_aligned = T_c1_c0 @ E0_aligned                     # w2c(cur frame1, global)

    # 5) 下一次对齐（比如 2-3 之后对齐 3-4）需要用“当前chunk最后一帧”的全局外参
    prev_overlap_for_next = to3x4(E1_aligned)

    return to3x4(E0_aligned), to3x4(E1_aligned), prev_overlap_for_next


# ---------------- main ----------------

model = DepthAnything3.from_pretrained(
    "/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3-LARGE-1.1"
).to("cuda")

viewer = SLAMViewer(port=8080)

image0 = "/home/zhouyi/repo/dataset/sydney/000000.png"
image1 = "/home/zhouyi/repo/dataset/sydney/000010.png"
image2 = "/home/zhouyi/repo/dataset/sydney/000020.png"
image3 = "/home/zhouyi/repo/dataset/sydney/000030.png"

# ---- chunkA [0,1]：定义初始坐标系并渲染 ----
chunkA = model.inference(image=[image0, image1], use_ray_pose=True)

img0 = image_to_chw01(chunkA, 0)
img1 = image_to_chw01(chunkA, 1)

viewer.add_frame(img0, chunkA.depth[0], chunkA.conf[0], chunkA.extrinsics[0], chunkA.intrinsics[0], 1)
viewer.add_frame(img1, chunkA.depth[1], chunkA.conf[1], chunkA.extrinsics[1], chunkA.intrinsics[1], 2)

# 初始：上一chunk overlap(最后一帧=frame1) 的“全局外参”就是它自己（因为全局=chunkA坐标）
prev_overlap_aligned = chunkA.extrinsics[-1]
prev_chunk = chunkA

time.sleep(3)

# ---- chunkB [1,2]：对齐到初始坐标系并渲染 ----
chunkB = model.inference(image=[image1, image2], use_ray_pose=True)

img1b = image_to_chw01(chunkB, 0)
img2b = image_to_chw01(chunkB, 1)

E0_b, E1_b, prev_overlap_aligned = get_aligned_two_frame_extrinsics(
    prev_overlap_aligned, prev_chunk, chunkB
)

viewer.add_frame(img1b, chunkB.depth[0], chunkB.conf[0], E0_b, chunkB.intrinsics[0], 3)
viewer.add_frame(img2b, chunkB.depth[1], chunkB.conf[1], E1_b, chunkB.intrinsics[1], 4)

prev_chunk = chunkB

time.sleep(3)

# ---- chunkC [2,3]：对齐到初始坐标系并渲染 ----
chunkC = model.inference(image=[image2, image3], use_ray_pose=True)

img2c = image_to_chw01(chunkC, 0)
img3c = image_to_chw01(chunkC, 1)

E0_c, E1_c, prev_overlap_aligned = get_aligned_two_frame_extrinsics(
    prev_overlap_aligned, prev_chunk, chunkC
)

viewer.add_frame(img2c, chunkC.depth[0], chunkC.conf[0], E0_c, chunkC.intrinsics[0], 5)
viewer.add_frame(img3c, chunkC.depth[1], chunkC.conf[1], E1_c, chunkC.intrinsics[1], 6)

prev_chunk = chunkC

print("SLAM running. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(0.01)
except KeyboardInterrupt:
    print("stopped by user")