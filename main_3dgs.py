from depth_anything_3.api import DepthAnything3
import numpy as np
import time

# ---------------- main ----------------

model = DepthAnything3.from_pretrained(
    # "/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3-LARGE-1.1"
    "/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3NESTED-GIANT-LARGE-1.1"
).to("cuda")

image0 = "/home/zhouyi/repo/dataset/sydney/000000.png"
image1 = "/home/zhouyi/repo/dataset/sydney/000010.png"
image2 = "/home/zhouyi/repo/dataset/sydney/000020.png"
image3 = "/home/zhouyi/repo/dataset/sydney/000030.png"
# image0 = "/home/zhouyi/repo/dataset/statue/frame_00007.png"
# image1 = "/home/zhouyi/repo/dataset/statue/frame_00017.png"
# image2 = "/home/zhouyi/repo/dataset/statue/frame_00028.png"
# image3 = "/home/zhouyi/repo/dataset/statue/frame_00044.png"

sleep_time = 10

prediction = model.inference(
                        image=[image0, image1],  
                        use_ray_pose=True,
                        infer_gs=True,  
                        export_dir = "/home/zhouyi/repo/DA3-SLAM/output",
                        export_format = "gs_ply"
                        )


time.sleep(sleep_time)
