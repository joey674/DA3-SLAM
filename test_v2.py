# test.py
# ------------------------------------------------------------
# Manual 3-chunk demo:
# - chunkA: [0,1] defines global coordinate system
# - chunkB: [1,2] aligned to global using Sim3 on overlap frame (image1)
# - chunkC: [2,3] aligned to global using Sim3 on overlap frame (image2)
# ------------------------------------------------------------

from depth_anything_3.api import DepthAnything3
import time

from viewer import SLAMViewer
from test_align_v2 import (
    image_to_chw01,
    get_aligned_two_frame_extrinsics_sim3,
)

model = DepthAnything3.from_pretrained(
    "/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3-LARGE-1.1"
).to("cuda")

viewer = SLAMViewer(port=8080)
time.sleep(1)

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

prev_overlap_aligned = chunkA.extrinsics[-1]  # image1 pose in global (since global==chunkA)
prev_chunk = chunkA

time.sleep(3)

# ---- chunkB [1,2]：对齐到初始坐标系并渲染 ----
chunkB = model.inference(image=[image1, image2], use_ray_pose=True)

img1b = image_to_chw01(chunkB, 0)
img2b = image_to_chw01(chunkB, 1)

E0_b, E1_b, prev_overlap_aligned, s_b = get_aligned_two_frame_extrinsics_sim3(
    prev_overlap_aligned, prev_chunk, chunkB, threshold=0.05, max_iterations=50
)
print("[chunkB] sim3 scale =", s_b)

viewer.add_frame(img1b, chunkB.depth[0], chunkB.conf[0], E0_b, chunkB.intrinsics[0], 3)
viewer.add_frame(img2b, chunkB.depth[1], chunkB.conf[1], E1_b, chunkB.intrinsics[1], 4)

prev_chunk = chunkB
time.sleep(3)

# ---- chunkC [2,3]：对齐到初始坐标系并渲染 ----
chunkC = model.inference(image=[image2, image3], use_ray_pose=True)

img2c = image_to_chw01(chunkC, 0)
img3c = image_to_chw01(chunkC, 1)

E0_c, E1_c, prev_overlap_aligned, s_c = get_aligned_two_frame_extrinsics_sim3(
    prev_overlap_aligned, prev_chunk, chunkC, threshold=0.05, max_iterations=50
)
print("[chunkC] sim3 scale =", s_c)

viewer.add_frame(img2c, chunkC.depth[0], chunkC.conf[0], E0_c, chunkC.intrinsics[0], 5)
viewer.add_frame(img3c, chunkC.depth[1], chunkC.conf[1], E1_c, chunkC.intrinsics[1], 6)

print("SLAM running. Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(0.01)
except KeyboardInterrupt:
    print("stopped by user")
