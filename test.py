from depth_anything_3.api import DepthAnything3
import numpy as np
import time

from viewer import SLAMViewer

# 你自己的对齐工具
from test_align import (
    get_aligned_two_frame_extrinsics,
    image_to_chw01,
    estimate_depth_scale
)

# ---------------- main ----------------

model = DepthAnything3.from_pretrained(
    "/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3-LARGE-1.1"
    # "/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3NESTED-GIANT-LARGE-1.1"
).to("cuda")

viewer = SLAMViewer(port=8080)

# image0 = "/home/zhouyi/repo/dataset/sydney/000000.png"
# image1 = "/home/zhouyi/repo/dataset/sydney/000010.png"
# image2 = "/home/zhouyi/repo/dataset/sydney/000020.png"
# image3 = "/home/zhouyi/repo/dataset/sydney/000030.png"
image0 = "/home/zhouyi/repo/dataset/statue/frame_00007.png"
image1 = "/home/zhouyi/repo/dataset/statue/frame_00017.png"
image2 = "/home/zhouyi/repo/dataset/statue/frame_00028.png"
image3 = "/home/zhouyi/repo/dataset/statue/frame_00044.png"

sleep_time = 10

# ---- chunkA [0,1]：定义初始坐标系并渲染 ----
chunkA = model.inference(image=[image0, image1], use_ray_pose=True)

img0 = image_to_chw01(chunkA, 0)
img1 = image_to_chw01(chunkA, 1)

viewer.add_frame(img0, chunkA.depth[0], chunkA.conf[0], chunkA.extrinsics[0], chunkA.intrinsics[0], 1)
viewer.add_frame(img1, chunkA.depth[1], chunkA.conf[1], chunkA.extrinsics[1], chunkA.intrinsics[1], 2)

# 初始：上一chunk overlap(最后一帧=frame1) 的“全局外参”就是它自己（因为全局=chunkA坐标）
prev_overlap_aligned = chunkA.extrinsics[-1]
prev_chunk = chunkA

time.sleep(sleep_time)

# ---- chunkB [1,2]：对齐到初始坐标系并渲染 ----
chunkB = model.inference(image=[image1, image2], use_ray_pose=True)

img1b = image_to_chw01(chunkB, 0)
img2b = image_to_chw01(chunkB, 1)

E0_b, E1_b, prev_overlap_aligned,s = get_aligned_two_frame_extrinsics(
    prev_overlap_aligned, prev_chunk, chunkB
)

#depth scale
s_depth = estimate_depth_scale(prev_chunk, chunkB, conf_th=0.2)
chunkB_depth_scaled = chunkB.depth * s_depth  
print("s_depth =", s_depth)

viewer.add_frame(img1b, chunkB_depth_scaled[0], chunkB.conf[0], E0_b, chunkB.intrinsics[0], 3)
viewer.add_frame(img2b, chunkB_depth_scaled[1], chunkB.conf[1], E1_b, chunkB.intrinsics[1], 4)

prev_chunk = chunkB

time.sleep(sleep_time)

# ---- chunkC [2,3]：对齐到初始坐标系并渲染 ----
chunkC = model.inference(image=[image2, image3], use_ray_pose=True)

img2c = image_to_chw01(chunkC, 0)
img3c = image_to_chw01(chunkC, 1)

E0_c, E1_c, prev_overlap_aligned,s = get_aligned_two_frame_extrinsics(
    prev_overlap_aligned, prev_chunk, chunkC
)

#depth scale
s_depth = estimate_depth_scale(prev_chunk, chunkC, conf_th=0.2)
chunkC_depth_scaled = chunkC.depth * s_depth  
print("s_depth =", s_depth)

viewer.add_frame(img2c, chunkC_depth_scaled[0], chunkC.conf[0], E0_c, chunkC.intrinsics[0], 5)
viewer.add_frame(img3c, chunkC_depth_scaled[1], chunkC.conf[1], E1_c, chunkC.intrinsics[1], 6)

prev_chunk = chunkC

print("SLAM running. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(0.01)
except KeyboardInterrupt:
    print("stopped by user")