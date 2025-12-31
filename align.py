"""
点云配准模块
提供通用的点云转换、对齐和配准功能
"""

import numpy as np
from typing import Tuple
from geometry import (
    depth_to_point_cloud_vectorized,
    apply_sim3_transform,
    )

# irls
def weighted_umeyama_alignment(src, dst, w):
    eps = 1e-8
    w = w.astype(np.float64)
    w = w / (np.sum(w) + eps)

    mu_src = np.sum(src * w[:, None], axis=0)
    mu_dst = np.sum(dst * w[:, None], axis=0)

    X = src - mu_src
    Y = dst - mu_dst

    Sigma_xy = (Y * w[:, None]).T @ X  # 3x3

    U, S, Vt = np.linalg.svd(Sigma_xy)

    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0

    R = U @ D @ Vt

    var_src = np.sum(w * np.sum(X * X, axis=1))

    s = float((S @ np.diag(D)) / (var_src + eps))   # S:(3,), diag(D):(3,)

    t = mu_dst - s * (R @ mu_src)
    return s, R, t

def weighted_umeyama_alignment0(points1: np.ndarray, points2: np.ndarray, 
                               weights: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    加权Umeyama算法 - 求解加权最小二乘的Sim(3)变换
    
    Args:
        points1: 源点云 [N, 3]
        points2: 目标点云 [N, 3]
        weights: 权重 [N]
        
    Returns:
        s: 尺度因子
        R: 旋转矩阵 [3, 3]
        t: 平移向量 [3]
    """
    # 确保权重和为1
    weights = weights / (np.sum(weights) + 1e-8)
    
    # 计算加权中心
    centroid1 = np.sum(points1 * weights[:, np.newaxis], axis=0)
    centroid2 = np.sum(points2 * weights[:, np.newaxis], axis=0)
    
    # 去中心化
    points1_centered = points1 - centroid1
    points2_centered = points2 - centroid2
    
    # 计算加权协方差矩阵
    W = np.diag(weights)
    S = np.dot(points2_centered.T, np.dot(W, points1_centered))
    
    # SVD分解
    U, Sigma, Vt = np.linalg.svd(S)
    
    # 计算旋转矩阵
    R = np.dot(U, Vt)
    
    # 确保旋转矩阵是右手系 (行列式为1 )
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)
    
    # 计算尺度因子
    trace_S = np.trace(S)
    scale_num = trace_S
    scale_den = np.sum(weights * np.sum(points1_centered ** 2, axis=1))
    s = scale_num / (scale_den + 1e-8)
    
    # 计算平移向量
    t = centroid2 - s * np.dot(R, centroid1)
    
    return s, R, t

def huber_weight(residual: float, delta: float = 1.0) -> float:
    """
    Huber损失函数的权重计算
    
    Args:
        residual: 残差
        delta: Huber损失参数
        
    Returns:
        权重值
    """
    abs_r = abs(residual)
    if abs_r <= delta:
        return 1.0
    else:
        return delta / abs_r

def align_two_point_clouds_irls(point_map1: np.ndarray, point_map2: np.ndarray,
                                conf1: np.ndarray, conf2: np.ndarray,
                                min_points: int = 100,
                                max_iterations: int = 20,
                                convergence_threshold: float = 1e-6) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    使用IRLS算法对齐两个点云 基于论文算法
    
    Args:
        point_map1: 第一个点云 [overlap_size, H, W, 3]
        point_map2: 第二个点云 [overlap_size, H, W, 3]
        conf1: 第一个点云的置信度 [overlap_size, H, W]
        conf2: 第二个点云的置信度 [overlap_size, H, W]
        min_points: 最小点数阈值
        max_iterations: 最大迭代次数
        convergence_threshold: 收敛阈值
        
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
    
    # 根据论文  直接丢弃置信度低于中位数0.1的点
    conf_threshold = min(np.median(confs1), np.median(confs2)) * 0.1
    print(f"  Align with confidence threshold: {conf_threshold}")
    
    mask1 = confs1 > conf_threshold
    mask2 = confs2 > conf_threshold
    
    points1_filtered = points1[mask1]
    points2_filtered = points2[mask2]
    confs1_filtered = confs1[mask1]
    confs2_filtered = confs2[mask2]
    
    # 如果点数不足 返回单位变换
    if len(points1_filtered) < min_points or len(points2_filtered) < min_points:
        print(f"  Warning: Not enough points for alignment: {len(points1_filtered)} vs {len(points2_filtered)}")
        return 1.0, np.eye(3), np.zeros(3)
    
    # 采样点以加速计算 (但保留对应关系 )
    sample_size = min(5000, len(points1_filtered), len(points2_filtered))
    indices = np.random.choice(len(points1_filtered), sample_size, replace=False)
    
    points1_sampled = points1_filtered[indices]
    points2_sampled = points2_filtered[indices]
    confs_sampled = np.sqrt(confs1_filtered[indices] * confs2_filtered[indices])  # 几何平均
    
    print(f"  Map1 has {len(points1_filtered)},map2 has {len(points1_filtered)}; Using {sample_size} points for IRLS alignment")
    
    # 初始化变换  单位变换
    s = 1.0
    R = np.eye(3)
    t = np.zeros(3)
    
    # IRLS迭代
    for iteration in range(max_iterations):
        # 计算当前变换下的残差
        points2_transformed = apply_sim3_transform(points2_sampled, s, R, t)
        residuals = np.linalg.norm(points1_sampled - points2_transformed, axis=1)
        
        # 根据公式(3)计算权重  w_i = c_i * ρ'(r_i) / r_i
        weights = np.zeros_like(residuals)
        for i in range(len(residuals)):
            r = residuals[i]
            c = confs_sampled[i]
            
            # Huber损失导数部分  ρ'(r)/r
            if r == 0:
                huber_weight_val = 1.0
            else:
                huber_weight_val = huber_weight(r)
            
            weights[i] = c * huber_weight_val
        
        # 归一化权重
        weights = weights / (np.max(weights) + 1e-8)
        
        # 使用加权Umeyama算法求解新的变换
        s_new, R_new, t_new = weighted_umeyama_alignment(points2_sampled, points1_sampled, weights)
        
        # 检查收敛性
        transform_change = np.abs(s_new - s) + np.linalg.norm(R_new - R) + np.linalg.norm(t_new - t)
        
        # 更新变换
        s, R, t = s_new, R_new, t_new
        
        print(f"    Iteration {iteration + 1}: s={s:.4f}, "
              f"avg_residual={np.mean(residuals):.4f}, "
              f"change={transform_change:.6f}")
        
        if transform_change < convergence_threshold:
            print(f"  IRLS converged after {iteration + 1} iterations")
            break
    
    # 计算最终对齐误差
    points2_transformed = apply_sim3_transform(points2_sampled, s, R, t)
    final_residuals = np.linalg.norm(points1_sampled - points2_transformed, axis=1)
    print(f"  Final alignment: s={s:.4f}, avg_error={np.mean(final_residuals):.4f}")
    
    return s, R, t


# align api
def align_two_point_clouds(point_map1: np.ndarray, point_map2: np.ndarray,
                          conf1: np.ndarray, conf2: np.ndarray,
                          min_points: int = 100) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    对齐两个点云 (兼容旧接口 )
    
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
    # 调用新的IRLS算法
    return align_two_point_clouds_irls(point_map1, point_map2, conf1, conf2, min_points)
    # return align_two_point_clouds_icp(point_map1, point_map2, conf1, conf2)



# help function
def extract_overlap_chunk_prediction(prev_chunk_prediction: dict, cur_chunk_prediction: dict, 
                                          overlap_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    提取两个块重叠区域的点云和置信度
    
    Args:
        chunk1_data: 第一个块的数据
        chunk2_data: 第二个块的数据
        overlap_size: 重叠帧数
        
    Returns:
        point_map1: 第一个块重叠区域的点云 [overlap_size, H, W, 3]
        point_map2: 第二个块重叠区域的点云 [overlap_size, H, W, 3]
        conf1: 第一个块重叠区域的置信度 [overlap_size, H, W]
        conf2: 第二个块重叠区域的置信度 [overlap_size, H, W]
    """
    # 第一个块的最后overlap_size帧
    point_map1 = depth_to_point_cloud_vectorized(
        prev_chunk_prediction['depth'][-overlap_size:],
        prev_chunk_prediction['intrinsics'][-overlap_size:],
        prev_chunk_prediction['extrinsics'][-overlap_size:]
    )
    
    # 第二个块的前overlap_size帧
    point_map2 = depth_to_point_cloud_vectorized(
        cur_chunk_prediction['depth'][:overlap_size],
        cur_chunk_prediction['intrinsics'][:overlap_size],
        cur_chunk_prediction['extrinsics'][:overlap_size]
    )
    
    # 提取对应的置信度
    conf1 = prev_chunk_prediction['conf'][-overlap_size:]
    conf2 = cur_chunk_prediction['conf'][:overlap_size]
    
    return point_map1, point_map2, conf1, conf2

