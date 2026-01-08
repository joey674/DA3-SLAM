# solver.py
import numpy as np
import torch
import time
from typing import List, Dict, Tuple
from collections import deque

from utils import (
    load_image,
    extract_keyframe,
)

# 改成新的单帧对齐辅助文件
from align_geometry_single import (
    get_aligned_chunk_extrinsics_single_overlap,
    image_to_chw01,
    estimate_depth_scale,
)


class SLAMSolver:
    def __init__(self,
                 image_dir,
                 config):
        self.config = config
        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap_size = self.config["Model"]["overlap_size"]  # 仍保留，但对齐按单帧使用
        self.image_dir = image_dir

        # 计数器
        self.chunk_count = 0

        # 数据存储 - 流式处理
        self.frame_buffer: deque = deque(maxlen=self.chunk_size * 2)
        self.chunk_prediction_list: List[Dict] = []
        
        # 与chunk对齐相关容器
        self.sim3_transforms: List[Tuple[float, np.ndarray, np.ndarray]] = []  # relative to prev chunk
        self.prev_overlap_aligned_3x4 = None # 记录“上一 chunk overlap(最后一帧)”在全局坐标系下的 w2c

        # 模型
        self.model = None
        self.load_model()

        # 可视化器
        self.viewer = None
        self.init_viewer()

    def load_model(self):
        """加载DA3模型"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        try:
            from depth_anything_3.api import DepthAnything3
            model_path = self.config["Weights"]["DA3"]
            print(f"Loading DA3 model from {model_path}...")
            self.model = DepthAnything3.from_pretrained(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")
        except ImportError as e:
            print(f"Failed to load DA3 model: {e}")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def init_viewer(self):
        """初始化可视化器"""
        port = self.config["Model"]["port"]
        try:
            from viewer import SLAMViewer
            self.viewer = SLAMViewer(port=port)
            print(f"Viewer initialized on port {port}")
        except ImportError as e:
            print(f"Failed to initialize viewer: {e}")
            self.viewer = None

    def update_buffer_after_chunk_processed(self):
        """处理完chunk后更新缓冲区"""
        if len(self.frame_buffer) > self.overlap_size:
            for _ in range(self.chunk_size - self.overlap_size):
                if self.frame_buffer:
                    self.frame_buffer.popleft()

    def update_viewer(self, chunk_prediction: Dict):
        """
        按 chunk_prediction 里的 extrinsics_global 渲染所有帧（包含 overlap 帧，行为与 main_align 一致）
        """
        if self.viewer is None:
            return

        extrinsics_global = chunk_prediction.get("extrinsics_global", None)
        if extrinsics_global is None:
            # fallback：如果没对齐字段，就用原始 extrinsics
            extrinsics_global = chunk_prediction["extrinsics"]

        n = len(chunk_prediction["image_paths"])
        for i in range(n):
            image_chw = image_to_chw01(chunk_prediction, i)
            depth = chunk_prediction["depth"][i]
            conf = chunk_prediction["conf"][i]
            intrinsic = chunk_prediction["intrinsics"][i]
            extrinsic_global = extrinsics_global[i]

            self.viewer.add_frame(
                image=image_chw,
                depth=depth,
                conf=conf,
                extrinsic=extrinsic_global,
                intrinsic=intrinsic,
            )

    def process_chunk_alignment(self, prev_chunk_prediction: Dict, cur_chunk_prediction: Dict) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        单帧 overlap 对齐：
          1) 用 estimate_depth_scale 做深度尺度对齐（乘到 cur_chunk.depth 上）
          2) 用 prev(last) vs cur(first) 的点云做 ICP 求 R,t
          3) 计算 cur chunk 所有帧的 extrinsics_global 相对于第一帧的世界坐标系
          4) 更新 self.prev_overlap_aligned_3x4 = cur chunk 最后一帧全局外参
        """
        # --- 先做尺度对齐（按 overlap 单帧）---
        s_depth = estimate_depth_scale(prev_chunk_prediction, cur_chunk_prediction, conf_th=0.2)
        cur_chunk_prediction["depth"] = cur_chunk_prediction["depth"] * s_depth

        # --- 计算当前 chunk 全局外参链 ---
        extrinsics_global, prev_overlap_for_next, (s, R, t) = get_aligned_chunk_extrinsics_single_overlap(
            prev_overlap_aligned_3x4=self.prev_overlap_aligned_3x4,
            prev_chunk_prediction=prev_chunk_prediction,
            cur_chunk_prediction=cur_chunk_prediction,
        )

        cur_chunk_prediction["extrinsics_global"] = extrinsics_global
        self.prev_overlap_aligned_3x4 = prev_overlap_for_next

        return s, R, t

    def run_single_chunk_prediction(self, chunk_image_paths: List[str]) -> Dict:
        """
        处理单个块
        """
        print(f"  Predict single chunk with {len(chunk_image_paths)} images through da3...")

        torch.cuda.empty_cache()
        with torch.no_grad():
            prediction = self.model.inference(
                image=chunk_image_paths,
                process_res_method="upper_bound_resize",
            )

            result = {
                "chunk_idx": self.chunk_count,
                "image_paths": chunk_image_paths,
                "processed_images": prediction.processed_images,  # [N,H,W,3] uint8
                "depth": prediction.depth,                        # [N,H,W]
                "conf": prediction.conf,                          # [N,H,W]
                "extrinsics": prediction.extrinsics,              # [N,3,4] w2c (local chunk coords)
                "intrinsics": prediction.intrinsics,              # [N,3,3]
            }
            return result

    def load_chunk_image_paths(self) -> List[str]:
        """
        从缓冲区准备chunk数据
        """
        chunk_frames = list(self.frame_buffer)[:self.chunk_size]
        return chunk_frames

    def should_run_chunk_prediction(self) -> bool:
        """
        判断是否应该处理当前chunk
        """
        return len(self.frame_buffer) >= self.chunk_size

    def process_frame(self, image_path: str):
        """
        处理一帧图像
        """
        self.frame_buffer.append(image_path)

        if self.should_run_chunk_prediction():
            print(f"\n  Processing chunk {self.chunk_count}...")

            chunk_image_paths = self.load_chunk_image_paths()
            cur_chunk_prediction = self.run_single_chunk_prediction(chunk_image_paths)
            self.chunk_prediction_list.append(cur_chunk_prediction)

            if self.chunk_count == 0:
                # 第一 chunk：定义为全局坐标系（global = local）
                cur_chunk_prediction["extrinsics_global"] = cur_chunk_prediction["extrinsics"]

                # 初始化 prev_overlap_aligned 为第一 chunk 最后一帧
                self.prev_overlap_aligned_3x4 = cur_chunk_prediction["extrinsics_global"][-1]

                # identity transform 占位，避免后面取 [-1] 崩
                self.sim3_transforms.append((1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)))
            else:
                prev_chunk_prediction = self.chunk_prediction_list[self.chunk_count - 1]
                s, R, t = self.process_chunk_alignment(prev_chunk_prediction, cur_chunk_prediction)
                self.sim3_transforms.append((s, R, t))

            # 更新可视化（用 extrinsics_global）
            self.update_viewer(cur_chunk_prediction)

            # 更新缓冲区
            self.update_buffer_after_chunk_processed()

            # chunk + 1
            self.chunk_count += 1

            time.sleep(self.config["Model"]["sleep_between_chunk"])
            print("  Sleep for observation")
            print("#" * 50)

    def run(self):
        print("=" * 50)
        print("Starting DA3-SLAM ...")
        print("=" * 50)

        image_paths = load_image(self.image_dir)
        if not image_paths:
            print(f"Warning: No images found in {self.image_dir}")
            return

        image_paths = extract_keyframe(image_paths, self.config["Model"]["keyframe_interval"])

        for img_path in image_paths:
            self.process_frame(img_path)

        print("=" * 50)
        print("SLAM process completed!")
        print("=" * 50)