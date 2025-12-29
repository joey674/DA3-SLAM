"""
点云配准模块
提供通用的点云转换、对齐和配准功能
"""

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


def extract_overlap_points(chunk1_data: dict, chunk2_data: dict, overlap_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取两个块重叠区域的点云
    
    Args:
        chunk1_data: 第一个块的数据，包含'depth', 'intrinsics', 'extrinsics'
        chunk2_data: 第二个块的数据，包含'depth', 'intrinsics', 'extrinsics'
        overlap_size: 重叠帧数
        
    Returns:
        point_map1: 第一个块重叠区域的点云 [overlap_size, H, W, 3]
        point_map2: 第二个块重叠区域的点云 [overlap_size, H, W, 3]
    """
    # 第一个块的最后overlap_size帧
    point_map1 = depth_to_point_cloud_vectorized(
        chunk1_data['depth'][-overlap_size:],
        chunk1_data['intrinsics'][-overlap_size:],
        chunk1_data['extrinsics'][-overlap_size:]
    )
    
    # 第二个块的前overlap_size帧
    point_map2 = depth_to_point_cloud_vectorized(
        chunk2_data['depth'][:overlap_size],
        chunk2_data['intrinsics'][:overlap_size],
        chunk2_data['extrinsics'][:overlap_size]
    )
    
    return point_map1, point_map2


def align_two_point_clouds(point_map1: np.ndarray, point_map2: np.ndarray,
                          conf1: np.ndarray, conf2: np.ndarray,
                          min_points: int = 100) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    对齐两个点云
    
    Args:
        point_map1: 第一个点云 [overlap_size, H, W, 3]
        point_map2: 第二个点云 [overlap_size, H, W, 3]
        conf1: 第一个点云的置信度 [overlap_size, H, W]
        conf2: 第二个点云的置信度 [overlap_size, H, W]
        min_points: 最小点数阈值
        
    Returns:
        s: 尺度因子
        R: 旋转矩阵 [3, 3]
        t: 平移向量 [3]
    """
    # 展平点云和置信度
    points1 = point_map1.reshape(-1, 3)
    points2 = point_map2.reshape(-1, 3)
    confs1 = conf1.reshape(-1)
    confs2 = conf2.reshape(-1)
    
    # 使用置信度阈值过滤点
    conf_threshold = min(np.median(confs1), np.median(confs2)) * 0.1
    mask1 = confs1 > conf_threshold
    mask2 = confs2 > conf_threshold
    
    points1_filtered = points1[mask1]
    points2_filtered = points2[mask2]
    
    # 如果点数不足，返回单位变换
    if len(points1_filtered) < min_points or len(points2_filtered) < min_points:
        print(f"  Warning: Not enough points for alignment: {len(points1_filtered)} vs {len(points2_filtered)}")
        return 1.0, np.eye(3), np.zeros(3)
    
    # 采样点以加速计算
    sample_size = min(1000, len(points1_filtered), len(points2_filtered))
    indices1 = np.random.choice(len(points1_filtered), sample_size, replace=False)
    indices2 = np.random.choice(len(points2_filtered), sample_size, replace=False)
    
    points1_sampled = points1_filtered[indices1]
    points2_sampled = points2_filtered[indices2]
    
    # 计算中心点
    center1 = np.mean(points1_sampled, axis=0)
    center2 = np.mean(points2_sampled, axis=0)
    
    # 去中心化
    points1_centered = points1_sampled - center1
    points2_centered = points2_sampled - center2
    
    # 使用SVD计算旋转矩阵
    H = np.dot(points1_centered.T, points2_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # 确保旋转矩阵是正交的（行列式为1）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # 计算尺度因子（简单的平均距离比）
    dist1 = np.linalg.norm(points1_centered, axis=1)
    dist2 = np.linalg.norm(points2_centered, axis=1)
    s = np.mean(dist2) / (np.mean(dist1) + 1e-8)
    
    # 计算平移
    t = center2 - s * np.dot(R, center1)
    
    print(f"  Estimated transform: s={s:.4f}, t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    
    return s, R, t


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


def filter_point_cloud(points: np.ndarray, colors: np.ndarray, confs: np.ndarray,
                       min_depth: float = 0.1, max_depth: float = 50.0,
                       min_confidence: float = 0.5, sample_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    过滤点云，移除无效点并采样
    
    Args:
        points: 点云 [N_points, 3]
        colors: 颜色 [N_points, 3]
        confs: 置信度 [N_points]
        min_depth: 最小深度
        max_depth: 最大深度
        min_confidence: 最小置信度
        sample_ratio: 采样比例
        
    Returns:
        filtered_points: 过滤后的点云
        filtered_colors: 过滤后的颜色
    """
    # 过滤无效点和低置信度点
    valid_mask = (
        (points[:, 2] > min_depth) &  # 深度大于最小值
        (points[:, 2] < max_depth) &  # 深度小于最大值
        np.all(np.isfinite(points), axis=1) &  # 所有坐标都是有限值
        (confs > min_confidence)  # 置信度大于阈值
    )
    
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    points_valid = points[valid_mask]
    colors_valid = colors[valid_mask]
    
    # 采样以减少点数
    sample_size = int(len(points_valid) * sample_ratio)
    if sample_size <= 0:
        return np.array([]), np.array([])
    
    indices = np.random.choice(len(points_valid), sample_size, replace=False)
    points_sampled = points_valid[indices]
    colors_sampled = colors_valid[indices]
    
    return points_sampled, colors_sampled


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