import torch
import numpy as np
from typing import Tuple
import open3d as o3d

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
    用 overlap 帧做 sim3，把当前 chunk 对齐到全局(初始)坐标系。
    prev_overlap_aligned_3x4: 上一个chunk overlap帧（最后一帧）的 w2c（已经是全局坐标系）
    返回：
      cur_frame0_aligned_3x4, cur_frame1_aligned_3x4, cur_overlap_aligned_lastframe_for_next
    """

    # 1) 取 overlap 点云（必须 camera coords）
    pc_prev, pc_cur = extract_overlap_point_cloud(prev_chunk_prediction, cur_chunk_prediction)
    pc_prev = pc_prev.reshape(-1, 3)
    pc_cur  = pc_cur.reshape(-1, 3)

    # 2) cur -> prev （2=>1）
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

    return to3x4(E0_aligned), to3x4(E1_aligned), prev_overlap_for_next,s

def estimate_depth_scale(prev_chunk, cur_chunk, conf_th=0.2, eps=1e-6):
    d_prev = prev_chunk.depth[-1]     # overlap frame in prev
    d_cur  = cur_chunk.depth[0]       # overlap frame in cur

    c_prev = prev_chunk.conf[-1] if hasattr(prev_chunk, "conf") else None
    c_cur  = cur_chunk.conf[0]  if hasattr(cur_chunk, "conf")  else None

    mask = (d_prev > eps) & (d_cur > eps) & np.isfinite(d_prev) & np.isfinite(d_cur)
    if c_prev is not None and c_cur is not None:
        mask &= (c_prev > conf_th) & (c_cur > conf_th)

    s_depth = np.median(d_prev[mask] / d_cur[mask])
    return float(s_depth)

# So3 ICP
def align_two_point_clouds_icp(source: np.ndarray, 
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

# Sim3 umeyama
def _umeyama_sim3(X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Solve: Y ≈ s R X + t
    X,Y: [N,3] with correspondences
    """
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    cov = (Yc.T @ Xc) / X.shape[0]
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    var_x = (Xc ** 2).sum() / X.shape[0]
    s = float((D * np.diag(S)).sum() / (var_x + 1e-12))

    t = mu_y - s * (R @ mu_x)
    return s, R, t

def align_two_point_clouds_umeyama(source: np.ndarray,
                                target: np.ndarray,
                                threshold: float = 0.001,
                                max_iterations: int = 30,
                                max_corr: int = 200000,
                                seed: int = 0) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Sim3-ICP:
      target ≈ s R source + t
    """
    # ---- clean ----
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]

    # ---- subsample (optional, keep it simple) ----
    rng = np.random.default_rng(seed)
    if source.shape[0] > max_corr:
        source = source[rng.choice(source.shape[0], max_corr, replace=False)]
    if target.shape[0] > max_corr:
        target = target[rng.choice(target.shape[0], max_corr, replace=False)]

    # Open3D KDTree on target
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target.astype(np.float64))
    kdtree = o3d.geometry.KDTreeFlann(tgt_pcd)

    # cumulative Sim3
    s_cum = 1.0
    R_cum = np.eye(3)
    t_cum = np.zeros(3)

    src = source.astype(np.float64)

    for _ in range(max_iterations):
        # transform source with current estimate
        src_warp = (s_cum * (R_cum @ src.T)).T + t_cum

        # nearest neighbor correspondences (src_warp -> target)
        idxs = np.empty(src_warp.shape[0], dtype=np.int32)
        d2 = np.empty(src_warp.shape[0], dtype=np.float64)

        for i, p in enumerate(src_warp):
            _, ind, dist2 = kdtree.search_knn_vector_3d(p, 1)
            idxs[i] = ind[0]
            d2[i] = dist2[0]

        # threshold filtering
        mask = d2 < (threshold * threshold)
        if mask.sum() < 20:
            break

        X = src[mask]                 # original source points (not warped)
        Y = target[idxs[mask]]        # matched target points

        # solve Sim3 between X and Y (in *global* sense)
        # BUT: we need to solve between current-warped source and Y, then compose.
        Xw = src_warp[mask]           # current-warped source points
        s_upd, R_upd, t_upd = _umeyama_sim3(Xw, Y)

        # compose: new_warp = s_upd R_upd (old_warp) + t_upd
        s_cum = s_upd * s_cum
        R_cum = R_upd @ R_cum
        t_cum = s_upd * (R_upd @ t_cum) + t_upd

    return float(s_cum), R_cum, t_cum

# api
def align_two_point_clouds(source: np.ndarray, 
                          target: np.ndarray,
                          threshold: float = 0.001,
                          max_iterations: int = 30) -> Tuple[float, np.ndarray, np.ndarray]:
    
    s,R,t = align_two_point_clouds_icp(source,target,threshold,max_iterations)
    # s,R,t = align_two_point_clouds_umeyama(source,target,threshold,max_iterations)
    print(f"s: {s}")
    print(f"R: {R}")
    print(f"t: {t}")
    return s,R,t

