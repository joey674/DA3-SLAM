import torch
import numpy as np
from typing import Tuple

def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None, in_coords = "world"):
    """
    depth: [N, H, W] numpy array or torch tensor
    intrinsics: [N, 3, 3] numpy array or torch tensor
    extrinsics: [N, 3, 4] (w2c) numpy array or torch tensor
    Returns: point_cloud: [N, H, W, 3] same type as input
    """
    assert in_coords in ("camera", "world")
    
    input_is_numpy = False
    if isinstance(depth, np.ndarray):
        input_is_numpy = True

        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)

        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intrinsics_tensor = intrinsics_tensor.to(device)
            extrinsics_tensor = extrinsics_tensor.to(device)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    # main logic

    N, H, W = depth_tensor.shape

    device = depth_tensor.device

    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)  # [N, 3, 3]
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    if in_coords == "camera":
        point_cloud = camera_coords
    else:  # in_coords == "world"
        c2w = torch.inverse(extrinsics_4x4)
        world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
        point_cloud = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud = point_cloud.cpu().numpy()
        
    return point_cloud


def extract_overlap_point_cloud(prev_chunk_prediction, cur_chunk_prediction) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取两个块重叠区域的点云和置信度
    
    Args:
        prev_chunk_prediction: 第一个块的数据 
        cur_chunk_prediction: 第二个块的数据
        
    Returns:
        point_map1: 第一个块重叠区域的点云 [overlap_size, H, W, 3]
        point_map2: 第二个块重叠区域的点云 [overlap_size, H, W, 3]
    """
    # 第一个块的最后overlap_size帧
    point_map1 = depth_to_point_cloud_vectorized(
        prev_chunk_prediction.depth[-1:],
        prev_chunk_prediction.intrinsics[-1:],
        prev_chunk_prediction.extrinsics[-1:],
        in_coords="camera"
    )
    
    # 第二个块的前overlap_size帧
    point_map2 = depth_to_point_cloud_vectorized(
        cur_chunk_prediction.depth[:1],
        cur_chunk_prediction.intrinsics[:1],
        cur_chunk_prediction.extrinsics[:1],
        in_coords="camera"
    )
    
    print(f"point_map1: {point_map1.shape}")
    print(f"point_map2: {point_map1.shape}")
    
    return point_map1, point_map2


def get_transformed_extrisinc(pre_chunk_extrinsics,s,R,t) -> np.ndarray:
    def to4x4(E3x4):
        E = np.eye(4)
        E[:3, :4] = E3x4
        return E
    T_c1_c2 = np.eye(4)
    T_c1_c2[:3,:3] = R
    T_c1_c2[:3, 3] = t
    E1 = to4x4(pre_chunk_extrinsics)   # w2c1
    E2_aligned = np.linalg.inv(T_c1_c2) @ E1           # w2c2（对齐后的）
    transformed_extrinsic = E2_aligned[:3, :4]
    return transformed_extrinsic

import open3d as o3d
def align_two_point_clouds(source: np.ndarray, 
                          target: np.ndarray,
                          threshold: float = 0.001,
                          max_iterations: int = 30) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    使用ICP算法配准两个点云
    
    参数:
        point_cloud1: 源点云 [N, 3]
        point_cloud2: 目标点云 [N, 3]
        threshold: ICP距离阈值
        max_iterations: 最大迭代次数
        
    返回:
        s: 尺度因子 (刚性变换默认为1.0)
        R: 3x3旋转矩阵
        t: 3x1平移向量
    """
    # 将numpy数组转换为Open3D点云
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    
    source_pcd.points = o3d.utility.Vector3dVector(source)
    target_pcd.points = o3d.utility.Vector3dVector(target)
    
    # 设置初始变换矩阵为单位矩阵
    trans_init = np.identity(4)
    
    # 执行ICP点云配准
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, 
        target_pcd, 
        threshold, 
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    
    # 提取变换矩阵
    transformation = reg_p2p.transformation
    
    # 提取旋转矩阵R和平移向量t
    # 注意：刚性变换的尺度因子s默认为1.0
    R = transformation[:3, :3]
    t = transformation[:3, 3]
    s = 1.0  # 点对点ICP假设是刚性变换
    
    return s, R, t