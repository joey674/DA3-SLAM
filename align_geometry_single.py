# align_geometry_single.py
import numpy as np
import torch
from typing import Tuple
import open3d as o3d


def to4x4(E3x4: np.ndarray) -> np.ndarray:
    E = np.eye(4, dtype=np.float64)
    E[:3, :4] = E3x4
    return E


def to3x4(E4x4: np.ndarray) -> np.ndarray:
    return E4x4[:3, :4]


def _get(pred, key: str):
    # pred can be dict or obj
    if isinstance(pred, dict):
        return pred[key]
    return getattr(pred, key)


def image_to_chw01(pred, idx: int) -> np.ndarray:
    imgs = _get(pred, "processed_images")  # [N,H,W,3] uint8
    img_hwc = imgs[idx]
    return img_hwc.transpose(2, 0, 1) / 255.0


def estimate_depth_scale(prev_chunk, cur_chunk, conf_th=0.2, eps=1e-6) -> float:
    d_prev = _get(prev_chunk, "depth")[-1]   # prev overlap = last frame
    d_cur  = _get(cur_chunk, "depth")[0]     # cur overlap  = first frame

    c_prev = _get(prev_chunk, "conf")[-1] if ("conf" in prev_chunk if isinstance(prev_chunk, dict) else hasattr(prev_chunk, "conf")) else None
    c_cur  = _get(cur_chunk, "conf")[0]  if ("conf" in cur_chunk  if isinstance(cur_chunk, dict)  else hasattr(cur_chunk,  "conf")) else None

    mask = (d_prev > eps) & (d_cur > eps) & np.isfinite(d_prev) & np.isfinite(d_cur)
    if c_prev is not None and c_cur is not None:
        mask &= (c_prev > conf_th) & (c_cur > conf_th)

    if mask.sum() < 50:
        # 太少就退化成 1
        return 1.0

    s_depth = np.median(d_prev[mask] / d_cur[mask])
    if not np.isfinite(s_depth) or s_depth <= 0:
        return 1.0
    return float(s_depth)


def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None, in_coords="camera"):
    """
    depth: [N,H,W] numpy or torch
    intrinsics: [N,3,3]
    extrinsics: [N,3,4] (w2c)  (这里只在 in_coords="world" 时会用到)
    """
    assert in_coords in ("camera", "world")

    input_is_numpy = isinstance(depth, np.ndarray)
    if input_is_numpy:
        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intr_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        ext_tensor = torch.tensor(extrinsics, dtype=torch.float32)
        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intr_tensor = intr_tensor.to(device)
            ext_tensor = ext_tensor.to(device)
    else:
        depth_tensor = depth
        intr_tensor = intrinsics
        ext_tensor = extrinsics
        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intr_tensor = intr_tensor.to(device)
            ext_tensor = ext_tensor.to(device)

    N, H, W = depth_tensor.shape
    device = depth_tensor.device

    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=device)
    pixel = torch.cat([u, v, ones], dim=-1)  # [N,H,W,3]

    intr_inv = torch.inverse(intr_tensor)  # [N,3,3]
    cam = torch.einsum("nij,nhwj->nhwi", intr_inv, pixel) * depth_tensor.unsqueeze(-1)  # [N,H,W,3]

    if in_coords == "camera":
        out = cam
    else:
        cam_h = torch.cat([cam, ones], dim=-1)  # [N,H,W,4]
        ext4 = torch.zeros((N, 4, 4), device=device)
        ext4[:, :3, :4] = ext_tensor
        ext4[:, 3, 3] = 1.0
        c2w = torch.inverse(ext4)
        world_h = torch.einsum("nij,nhwj->nhwi", c2w, cam_h)
        out = world_h[..., :3]

    if input_is_numpy:
        out = out.cpu().numpy()
    return out


def extract_single_overlap_point_cloud(prev_chunk_prediction, cur_chunk_prediction) -> Tuple[np.ndarray, np.ndarray]:
    """
    单帧 overlap：
      pc_prev: prev chunk last frame in CAMERA coords
      pc_cur : cur chunk first frame in CAMERA coords
    """
    d_prev = _get(prev_chunk_prediction, "depth")[-1:]
    k_prev = _get(prev_chunk_prediction, "intrinsics")[-1:]
    e_prev = _get(prev_chunk_prediction, "extrinsics")[-1:]

    d_cur = _get(cur_chunk_prediction, "depth")[:1]
    k_cur = _get(cur_chunk_prediction, "intrinsics")[:1]
    e_cur = _get(cur_chunk_prediction, "extrinsics")[:1]

    pc_prev = depth_to_point_cloud_vectorized(d_prev, k_prev, e_prev, in_coords="camera")
    pc_cur  = depth_to_point_cloud_vectorized(d_cur,  k_cur,  e_cur,  in_coords="camera")

    return pc_prev, pc_cur


# ---------- ICP alignment (SE3) ----------
def align_two_point_clouds_icp(source: np.ndarray,
                               target: np.ndarray,
                               threshold: float = 0.001,
                               max_iterations: int = 50) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    target ≈ R * source + t
    return s(=1), R, t
    """
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]

    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source.astype(np.float64))
    target_pcd.points = o3d.utility.Vector3dVector(target.astype(np.float64))

    trans_init = np.eye(4, dtype=np.float64)

    reg = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )

    T = reg.transformation
    R = T[:3, :3]
    t = T[:3, 3]
    s = 1.0
    return s, R, t


def align_two_point_clouds(source: np.ndarray,
                           target: np.ndarray,
                           threshold: float = 0.001,
                           max_iterations: int = 50) -> Tuple[float, np.ndarray, np.ndarray]:
    s, R, t = align_two_point_clouds_icp(source, target, threshold, max_iterations)
    return s, R, t


# ---------- core: align a chunk by single overlap frame ----------
def get_aligned_chunk_extrinsics_single_overlap(prev_overlap_aligned_3x4: np.ndarray,
                                                prev_chunk_prediction,
                                                cur_chunk_prediction,
                                                icp_threshold: float = 0.0001,
                                                icp_max_iter: int = 50):
    """
    单帧 overlap 对齐，把 cur chunk 对齐到“全局坐标系”（prev_overlap_aligned_3x4 所在的全局）。

    Inputs:
      prev_overlap_aligned_3x4:
        上一个 chunk 的最后一帧 w2c（已在全局系）
      prev_chunk_prediction:
        上一个 chunk（用于取最后帧深度点云）
      cur_chunk_prediction:
        当前 chunk（用于取第一帧点云 + chunk 内相对外参链）

    Returns:
      extrinsics_global: [N,3,4] 当前 chunk 每帧的全局 w2c
      prev_overlap_for_next: [3,4] 当前 chunk 最后一帧的全局 w2c
      (s,R,t): overlap ICP 得到的相对变换（cur->prev）
    """
    if prev_overlap_aligned_3x4 is None:
        raise ValueError("prev_overlap_aligned_3x4 is None. You must initialize it from the first chunk.")

    # 1) overlap 点云（camera coords）
    pc_prev, pc_cur = extract_single_overlap_point_cloud(prev_chunk_prediction, cur_chunk_prediction)
    pc_prev = pc_prev.reshape(-1, 3)
    pc_cur  = pc_cur.reshape(-1, 3)

    # 2) ICP：cur -> prev（target ≈ R*source + t）
    s, R, t = align_two_point_clouds(pc_cur, pc_prev, threshold=icp_threshold, max_iterations=icp_max_iter)

    T_prev_cur = np.eye(4, dtype=np.float64)
    T_prev_cur[:3, :3] = R
    T_prev_cur[:3, 3] = t

    # 3) 由 prev overlap 全局外参推 cur frame0 全局外参：
    #    E0_global = inv(T_prev_cur) @ E_prev_global
    E_prev_global = to4x4(prev_overlap_aligned_3x4)   # w2c(prev overlap, global)
    E0_global = np.linalg.inv(T_prev_cur) @ E_prev_global

    # 4) 用 chunk 内相对外参链推后续帧
    E_local = _get(cur_chunk_prediction, "extrinsics")  # [N,3,4] w2c(local)
    N = E_local.shape[0]

    extrinsics_global = np.zeros((N, 3, 4), dtype=np.float64)
    extrinsics_global[0] = to3x4(E0_global)

    E_prev_g = E0_global
    for i in range(1, N):
        Ei_local = to4x4(E_local[i])
        Eim1_local = to4x4(E_local[i - 1])

        # 相对：c(i-1) -> c(i)（w2c 形式）
        T_ci_cim1 = Ei_local @ np.linalg.inv(Eim1_local)

        Ei_global = T_ci_cim1 @ E_prev_g
        extrinsics_global[i] = to3x4(Ei_global)
        E_prev_g = Ei_global

    prev_overlap_for_next = extrinsics_global[-1]
    
    
    return extrinsics_global, prev_overlap_for_next, (float(s), R.astype(np.float64), t.astype(np.float64))