import numpy as np
from typing import Tuple, Optional,List

def depth_to_point_cloud_vectorized(depth: np.ndarray, intrinsics: np.ndarray, 
                                    extrinsics: np.ndarray) -> np.ndarray:
    """
    批量将深度图转换为点云
    
    Args:
        depth: [N, H, W] numpy array
        intrinsics: [N, 3, 3] numpy array
        extrinsics: [N, 3, 4] numpy array (w2c)
        
    Returns:
        point_cloud_world: [N, H, W, 3] numpy array
    """
    N, H, W = depth.shape
    
    # 创建像素坐标网格
    u = np.arange(W).reshape(1, 1, W, 1).repeat(N, axis=0).repeat(H, axis=1)
    v = np.arange(H).reshape(1, H, 1, 1).repeat(N, axis=0).repeat(W, axis=2)
    ones = np.ones((N, H, W, 1))
    pixel_coords = np.concatenate([u, v, ones], axis=-1)  # [N, H, W, 3]
    
    # 相机坐标系坐标
    intrinsics_inv = np.linalg.inv(intrinsics)  # [N, 3, 3]
    camera_coords = np.einsum('nij,nhwj->nhwi', intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth[..., np.newaxis]
    camera_coords_homo = np.concatenate([camera_coords, ones], axis=-1)  # [N, H, W, 4]
    
    # 转换为世界坐标系
    extrinsics_4x4 = np.zeros((N, 4, 4))
    extrinsics_4x4[:, :3, :4] = extrinsics
    extrinsics_4x4[:, 3, 3] = 1.0
    
    c2w = np.linalg.inv(extrinsics_4x4)  # 相机到世界变换
    world_coords_homo = np.einsum('nij,nhwj->nhwi', c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]
    
    return point_cloud_world


def apply_sim3_transform(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    应用SIM(3)变换到点云
    
    Args:
        points: 点云 [N, H, W, 3] 或 [N_points, 3]
        s: 尺度因子
        R: 旋转矩阵 [3, 3]
        t: 平移向量 [3]
        
    Returns:
        变换后的点云
    """
    if points.ndim == 4:
        # 批量点云 [N, H, W, 3]
        transformed = np.einsum('ij,...j->...i', R, points)  # 旋转
        transformed = s * transformed  # 尺度
        transformed = transformed + t  # 平移
    else:
        # 展平点云 [N_points, 3]
        transformed = np.dot(points, R.T)  # 旋转
        transformed = s * transformed  # 尺度
        transformed = transformed + t  # 平移
    
    return transformed


def accumulate_sim3_transforms(sim3_transforms: List[Tuple[float, np.ndarray, np.ndarray]]) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """
    累积SIM(3)变换，使每个变换都是相对于第一个块的
    
    Args:
        sim3_transforms: SIM(3)变换列表 [(s, R, t), ...]
        
    Returns:
        累积后的变换列表
    """
    if not sim3_transforms:
        return []
    
    accumulated = []
    current_s, current_R, current_t = 1.0, np.eye(3), np.zeros(3)
    
    # 第一个chunk没有变换
    accumulated.append((current_s, current_R, current_t))
    
    # 累积后续变换
    for s, R, t in sim3_transforms:
        # 累积尺度
        current_s = current_s * s
        
        # 累积旋转和平移
        current_t = np.dot(current_R, s * current_t) + t
        current_R = np.dot(R, current_R)
        
        accumulated.append((current_s, current_R, current_t))
    
    return accumulated


def transform_camera_pose(extrinsic: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    变换相机位姿
    
    Args:
        extrinsic: 原始外参矩阵 [3, 4] (w2c)
        s: 尺度因子
        R: 旋转矩阵 [3, 3]
        t: 平移向量 [3]
        
    Returns:
        变换后的外参矩阵 [3, 4] (w2c)
    """
    # 将w2c转换为c2w
    w2c_original = np.eye(4)
    w2c_original[:3, :4] = extrinsic
    c2w_original = np.linalg.inv(w2c_original)
    
    # 应用全局变换
    c2w_global = np.eye(4)
    c2w_global[:3, :3] = s * np.dot(R, c2w_original[:3, :3])
    c2w_global[:3, 3] = np.dot(R, s * c2w_original[:3, 3]) + t
    
    # 转换回w2c
    w2c_global = np.linalg.inv(c2w_global)
    extrinsic_global = w2c_global[:3, :4]
    
    return extrinsic_global

