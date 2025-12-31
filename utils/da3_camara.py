import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3-BASE")
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Load sample images
image_paths = [
    "/home/zhouyi/repo/dataset/sydney/000000.png",
    "/home/zhouyi/repo/dataset/sydney/000010.png"
]

# Step 1: 处理第一帧 获取外参
print("Step 1: Processing first frame to get extrinsics...")
prediction_single = model.inference(
    image=[image_paths[0]],  # 只处理第一帧
    process_res=504,
    process_res_method="upper_bound_resize",
    export_dir=None,
    export_format="mini_npz"  # 使用mini_npz格式获取完整预测
)

# 获取第一帧的外参
if prediction_single.extrinsics is not None:
    extrinsics_frame0 = prediction_single.extrinsics[0]  # 第一帧的外参
    print(f"First frame extrinsics shape: {extrinsics_frame0.shape}")
    print(f"First frame extrinsics:\n{extrinsics_frame0}")
else:
    print("No extrinsics estimated for first frame")
    extrinsics_frame0 = None

# Step 2: 两帧一起处理 传入第一帧的外参
print("\nStep 2: Processing both frames with first frame extrinsics...")

# 构建外参数组  第一帧使用已知外参 第二帧设为None (让模型估计 )
# 注意  inference方法期望所有帧都有外参 所以我们提供一个初始值
if extrinsics_frame0 is not None:
    extrinsics_frame1 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)
    # 创建一个初始外参数组  第一帧用已知值 第二帧用单位矩阵 (或零矩阵 )作为初始值
    extrinsics_input = np.array([
        extrinsics_frame0,  # 第一帧已知外参
        extrinsics_frame1
    ])
    
    print(f"Input extrinsics shape: {extrinsics_input.shape}")
    
    # 运行推理 传入外参
    prediction_both = model.inference(
        image=image_paths,
        extrinsics=extrinsics_input,  # 传入已知的外参
        align_to_input_ext_scale=True,  # 根据输入外参调整尺度
        process_res=504,
        process_res_method="upper_bound_resize",
        export_dir=None,
        export_format="mini_npz"
    )
    
    # 输出结果
    if prediction_both.extrinsics is not None:
        print(f"\nEstimated extrinsics shape: {prediction_both.extrinsics.shape}")
        print(f"\nFirst frame extrinsics (should be close to input):")
        print(prediction_both.extrinsics[0])
        print(f"\nSecond frame extrinsics (estimated by model):")
        print(prediction_both.extrinsics[1])
        
        # 计算相对位姿变化
        if extrinsics_frame0 is not None:
            # 第一帧输入外参与估计外参的差异
            diff_frame0 = np.linalg.norm(prediction_both.extrinsics[0] - extrinsics_frame0)
            print(f"\nDifference between input and estimated extrinsics for frame 0: {diff_frame0:.6f}")
            
            # 计算两帧之间的相对变换
            rel_transform = np.linalg.inv(prediction_both.extrinsics[0]) @ prediction_both.extrinsics[1]
            print(f"\nRelative transform from frame 0 to frame 1:")
            print(rel_transform)
            
            # 提取平移和旋转信息
            translation = rel_transform[:3, 3]
            rotation = rel_transform[:3, :3]
            print(f"\nTranslation vector: {translation}")
            print(f"Translation norm (movement magnitude): {np.linalg.norm(translation):.4f}")
    else:
        print("No extrinsics estimated in batch processing")
else:
    print("Cannot proceed without first frame extrinsics")

# Step 3: 可视化深度图进行比较
print("\nStep 3: Visualizing depth maps...")

# 获取第一帧单独处理时的深度图
if hasattr(prediction_single, 'depth') and prediction_single.depth is not None:
    depth_single = prediction_single.depth[0]
    print(f"Single frame depth shape: {depth_single.shape}")
    
    # 获取批量处理时的深度图
    if hasattr(prediction_both, 'depth') and prediction_both.depth is not None:
        depth_batch_frame0 = prediction_both.depth[0]
        depth_batch_frame1 = prediction_both.depth[1]
        print(f"Batch depth shapes: {depth_batch_frame0.shape}, {depth_batch_frame1.shape}")
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 显示原始图像
        img0 = Image.open(image_paths[0])
        img1 = Image.open(image_paths[1])
        
        axes[0, 0].imshow(img0)
        axes[0, 0].set_title("Frame 0 - Original")
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(img1)
        axes[1, 0].set_title("Frame 1 - Original")
        axes[1, 0].axis('off')
        
        # 显示深度图
        if len(depth_single.shape) == 2:
            axes[0, 1].imshow(depth_single, cmap='viridis')
            axes[0, 1].set_title("Frame 0 Depth (Single)")
            axes[0, 1].axis('off')
        else:
            # 如果是3通道 可能是可视化深度图
            axes[0, 1].imshow(depth_single)
            axes[0, 1].set_title("Frame 0 Depth (Single)")
            axes[0, 1].axis('off')
        
        if len(depth_batch_frame1.shape) == 2:
            axes[1, 1].imshow(depth_batch_frame1, cmap='viridis')
            axes[1, 1].set_title("Frame 1 Depth (Batch)")
            axes[1, 1].axis('off')
        else:
            axes[1, 1].imshow(depth_batch_frame1)
            axes[1, 1].set_title("Frame 1 Depth (Batch)")
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig("depth_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("Depth comparison saved as 'depth_comparison.png'")