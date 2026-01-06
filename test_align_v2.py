# test_align.py
# ------------------------------------------------------------
# Align overlapping Depth-Anything-3 chunks into one global coordinate system.
# - viewer.add_frame expects extrinsic: (3,4) w2c (world->camera)
# - Overlap alignment is done in CAMERA coordinates.
# - Provide both SE3(SO3) ICP and Sim3 (scale+R+t) alignment.
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple, Optional

import open3d as o3d


# --------------------------
# Basic helpers
# --------------------------

def to4x4(E3x4: np.ndarray) -> np.ndarray:
    """(3,4) -> (4,4)"""
    E = np.eye(4, dtype=np.float64)
    E[:3, :4] = E3x4
    return E


def to3x4(E4x4: np.ndarray) -> np.ndarray:
    """(4,4) -> (3,4)"""
    return E4x4[:3, :4]


def image_to_chw01(pred, idx: int) -> np.ndarray:
    """pred.processed_images[idx] (H,W,3 uint8) -> (3,H,W) float in [0,1]"""
    img_hwc = pred.processed_images[idx]
    return img_hwc.transpose(2, 0, 1) / 255.0


def clean_camera_points(pts: np.ndarray,
                        z_min: float = 0.1,
                        z_max: float = 50.0) -> np.ndarray:
    """
    Keep finite points and reasonable depth range in CAMERA coords.
    Assumes camera forward is +Z (typical pinhole).
    """
    if pts is None or pts.size == 0:
        return pts
    m = np.isfinite(pts).all(axis=1)
    m &= (pts[:, 2] > z_min) & (pts[:, 2] < z_max)
    return pts[m]


def maybe_subsample(pts: np.ndarray, max_points: int = 200_000, seed: int = 0) -> np.ndarray:
    """Randomly subsample to limit ICP cost."""
    if pts.shape[0] <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=max_points, replace=False)
    return pts[idx]


# --------------------------
# Depth -> point cloud
# --------------------------

def depth_to_point_cloud_vectorized(depth,
                                    intrinsics,
                                    extrinsics,
                                    device: Optional[str] = None,
                                    in_coords: str = "world") -> np.ndarray:
    """
    depth:      [N, H, W] (numpy or torch)
    intrinsics: [N, 3, 3]
    extrinsics: [N, 3, 4]  (w2c)
    in_coords:  "camera" or "world"
    Returns:    point_cloud [N, H, W, 3]  **always numpy** (so Open3D can consume)

    Note:
    - If in_coords == "camera": returns camera coordinates X_c
    - If in_coords == "world" : returns world coordinates X_w = c2w * X_c
    """
    assert in_coords in ("camera", "world")

    # convert inputs to torch tensors
    if isinstance(depth, np.ndarray):
        depth_t = torch.from_numpy(depth).float()
    else:
        depth_t = depth.float()

    if isinstance(intrinsics, np.ndarray):
        K_t = torch.from_numpy(intrinsics).float()
    else:
        K_t = intrinsics.float()

    if isinstance(extrinsics, np.ndarray):
        E_t = torch.from_numpy(extrinsics).float()
    else:
        E_t = extrinsics.float()

    if device is not None:
        depth_t = depth_t.to(device)
        K_t = K_t.to(device)
        E_t = E_t.to(device)

    N, H, W = depth_t.shape
    dev = depth_t.device

    u = torch.arange(W, device=dev).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=dev).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=dev)
    pix = torch.cat([u, v, ones], dim=-1)  # [N,H,W,3]

    K_inv = torch.inverse(K_t)  # [N,3,3]
    cam = torch.einsum("nij,nhwj->nhwi", K_inv, pix)  # [N,H,W,3]
    cam = cam * depth_t.unsqueeze(-1)

    if in_coords == "camera":
        out = cam
    else:
        # extrinsics is w2c, so c2w = inv(w2c)
        cam_h = torch.cat([cam, ones], dim=-1)  # [N,H,W,4]
        w2c = torch.zeros((N, 4, 4), device=dev)
        w2c[:, :3, :4] = E_t
        w2c[:, 3, 3] = 1.0
        c2w = torch.inverse(w2c)
        world_h = torch.einsum("nij,nhwj->nhwi", c2w, cam_h)
        out = world_h[..., :3]

    # IMPORTANT: always return numpy (Open3D can't take torch)
    return out.detach().cpu().numpy()


def extract_overlap_point_cloud(prev_chunk_prediction, cur_chunk_prediction) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlap definition for 2-frame chunks:
    - prev chunk: last frame  (-1)
    - cur  chunk: first frame (0)

    Returns point_map_prev, point_map_cur as [1,H,W,3] in CAMERA coords (numpy).
    """
    dev = "cuda" if torch.cuda.is_available() else None

    pc_prev = depth_to_point_cloud_vectorized(
        prev_chunk_prediction.depth[-1:],
        prev_chunk_prediction.intrinsics[-1:],
        prev_chunk_prediction.extrinsics[-1:],
        device=dev,
        in_coords="camera",
    )

    pc_cur = depth_to_point_cloud_vectorized(
        cur_chunk_prediction.depth[:1],
        cur_chunk_prediction.intrinsics[:1],
        cur_chunk_prediction.extrinsics[:1],
        device=dev,
        in_coords="camera",
    )

    return pc_prev, pc_cur


# --------------------------
# SO3/SE3 ICP (keep this!)
# --------------------------

def align_two_point_clouds_so3(source: np.ndarray,
                               target: np.ndarray,
                               threshold: float = 0.05,
                               max_iterations: int = 50,
                               max_points: int = 200_000,
                               seed: int = 0) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    SE3 (rigid) point-to-point ICP using Open3D.
    Returns: (s=1.0, R, t) such that target ≈ R*source + t
    """
    source = clean_camera_points(source)
    target = clean_camera_points(target)
    source = maybe_subsample(source, max_points=max_points, seed=seed)
    target = maybe_subsample(target, max_points=max_points, seed=seed + 1)

    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source.astype(np.float64))
    tgt.points = o3d.utility.Vector3dVector(target.astype(np.float64))

    trans_init = np.eye(4)
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )

    T = reg.transformation
    R = T[:3, :3]
    t = T[:3, 3]
    s = 1.0
    return s, R, t


# --------------------------
# Sim3 (scale + rotation + translation)
# --------------------------

def umeyama_sim3(X: np.ndarray, Y: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Closed-form Sim3 alignment (Umeyama).
    Solve: Y ≈ s R X + t

    X, Y: [N,3] assumed (approximately) corresponding.
    Returns s, R(3,3), t(3,)
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    n = X.shape[0]
    if n < 3:
        return 1.0, np.eye(3), np.zeros(3)

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    Xc = X - mu_x
    Yc = Y - mu_y

    cov = (Yc.T @ Xc) / n  # 3x3

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_x = (Xc ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / (var_x + 1e-12))
    else:
        s = 1.0

    t = mu_y - s * (R @ mu_x)
    return float(s), R, t


def align_two_point_clouds_sim3(source: np.ndarray,
                                target: np.ndarray,
                                threshold: float = 0.05,
                                max_iterations: int = 50,
                                max_points: int = 200_000,
                                seed: int = 0,
                                umeyama_points: int = 80_000) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Practical Sim3 alignment:
    1) run SE3 ICP to get a good rough alignment
    2) apply that SE3 to source
    3) estimate Sim3 (s,R,t) via Umeyama on a (subsampled) set of points
       (NOTE: correspondence is only approximate; good enough to fix scale drift quickly)

    Returns Sim3: target ≈ s R source + t
    """
    source = clean_camera_points(source)
    target = clean_camera_points(target)

    source = maybe_subsample(source, max_points=max_points, seed=seed)
    target = maybe_subsample(target, max_points=max_points, seed=seed + 1)

    # Step 1: SE3 ICP (rough)
    _, R0, t0 = align_two_point_clouds_so3(
        source, target, threshold=threshold, max_iterations=max_iterations,
        max_points=max_points, seed=seed
    )
    source_se3 = (R0 @ source.T).T + t0

    # Step 2: Umeyama Sim3 (refine including scale)
    X = maybe_subsample(source_se3, max_points=min(umeyama_points, source_se3.shape[0]), seed=seed + 2)
    Y = maybe_subsample(target,     max_points=min(umeyama_points, target.shape[0]),     seed=seed + 3)

    n = min(X.shape[0], Y.shape[0])
    X = X[:n]
    Y = Y[:n]

    s1, R1, t1 = umeyama_sim3(X, Y, with_scale=True)

    # Compose: target ≈ s1 R1 (R0 source + t0) + t1
    s = s1
    R = (R1 @ R0)
    t = (s1 * (R1 @ t0) + t1)
    return float(s), R, t


def sim3_inv(s: float, R: np.ndarray, t: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Inverse of Sim3:
      y = s R x + t
    => x = (1/s) R^T (y - t)
    """
    s_inv = 1.0 / float(s)
    R_inv = R.T
    t_inv = -s_inv * (R_inv @ t)
    return float(s_inv), R_inv, t_inv


def sim3_to_4x4(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Pack Sim3 into a "4x4-like" matrix for convenient left-multiply of w2c.
    We embed scale into the 3x3 block as (s*R).
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = float(s) * R
    T[:3, 3] = t
    return T


# --------------------------
# Extrinsic transformation utilities
# --------------------------

def get_transformed_extrinsic_so3(prev_overlap_aligned_w2c_3x4: np.ndarray,
                                  R: np.ndarray,
                                  t: np.ndarray) -> np.ndarray:
    """
    Given rigid ICP result (cur->prev): x_prev = R x_cur + t
    Update current overlap w2c in global:
      E_cur_aligned = inv(T_prev_cur) @ E_prev_aligned
    """
    T_prev_cur = np.eye(4, dtype=np.float64)
    T_prev_cur[:3, :3] = R
    T_prev_cur[:3, 3] = t

    E_prev = to4x4(prev_overlap_aligned_w2c_3x4)
    E_cur = np.linalg.inv(T_prev_cur) @ E_prev
    return to3x4(E_cur)


def get_transformed_extrinsic_sim3(prev_overlap_aligned_w2c_3x4: np.ndarray,
                                   s: float,
                                   R: np.ndarray,
                                   t: np.ndarray) -> np.ndarray:
    """
    Given Sim3 result (cur->prev): x_prev = s R x_cur + t
    Update current overlap w2c in global using Sim3 inverse:
      E_cur_aligned = inv_sim3(T_prev_cur) @ E_prev_aligned
    """
    s_inv, R_inv, t_inv = sim3_inv(s, R, t)
    T_prev_to_cur = sim3_to_4x4(s_inv, R_inv, t_inv)  # (prev->cur)

    E_prev = to4x4(prev_overlap_aligned_w2c_3x4)
    E_cur = T_prev_to_cur @ E_prev
    return to3x4(E_cur)


# --------------------------
# Chunk alignment (2-frame chunks)
# --------------------------

def get_aligned_two_frame_extrinsics_sim3(prev_overlap_aligned_w2c_3x4: np.ndarray,
                                          prev_chunk_prediction,
                                          cur_chunk_prediction,
                                          threshold: float = 0.05,
                                          max_iterations: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Align current 2-frame chunk to global using Sim3 (fix scale drift).
    Returns:
      E0_aligned_w2c_3x4, E1_aligned_w2c_3x4, prev_overlap_for_next (=E1_aligned), s_est

    Notes:
    - We align only the overlap frame via Sim3, then propagate to second frame
      using the chunk-local relative SE3 (often sufficient).
    """
    pc_prev, pc_cur = extract_overlap_point_cloud(prev_chunk_prediction, cur_chunk_prediction)
    pc_prev = pc_prev.reshape(-1, 3)
    pc_cur = pc_cur.reshape(-1, 3)

    s, R, t = align_two_point_clouds_sim3(pc_cur, pc_prev, threshold=threshold, max_iterations=max_iterations)

    # aligned overlap (frame0) with Sim3 inverse update
    E0_aligned = get_transformed_extrinsic_sim3(prev_overlap_aligned_w2c_3x4, s, R, t)

    # propagate to frame1 using local relative pose (SE3)
    E0_local = to4x4(cur_chunk_prediction.extrinsics[0])  # w2c local
    E1_local = to4x4(cur_chunk_prediction.extrinsics[1])  # w2c local
    T_c1_c0 = E1_local @ np.linalg.inv(E0_local)          # w2c-relative

    E0a = to4x4(E0_aligned)
    E1a = T_c1_c0 @ E0a

    E0a_3x4 = to3x4(E0a)
    E1a_3x4 = to3x4(E1a)
    return E0a_3x4, E1a_3x4, E1a_3x4, float(s)
