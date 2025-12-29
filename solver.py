# solver.py
import os
import numpy as np
import torch
import glob
import time
from typing import List, Dict, Tuple
from collections import deque

# 导入点云配准模块
from align import (
    depth_to_point_cloud_vectorized,
    extract_overlap_points,
    align_two_point_clouds,
    apply_sim3_transform,
    accumulate_sim3_transforms,
    filter_point_cloud,
    transform_camera_pose
)


class SLAMSolver:
    def __init__(self, 
                 viewer_port: int = 8080,
                 chunk_size: int = 30,
                 overlap_size: int = 15,
                 min_confidence: float = 0.5):
        """
        初始化SLAM求解器
        
        Args:
            viewer_port: 可视化端口
            chunk_size: 每组帧数
            overlap_size: 重叠帧数
            min_confidence: 最小置信度阈值
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_confidence = min_confidence
        
        # 帧计数器
        self.frame_count = 0
        self.chunk_count = 0
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 数据存储 - 流式处理
        self.frame_buffer: deque = deque(maxlen=chunk_size * 2)  # 帧缓冲区
        self.chunk_data_list: List[Dict] = []  # 已处理chunk数据
        self.sim3_transforms: List[Tuple[float, np.ndarray, np.ndarray]] = []  # SIM(3)变换列表
        self.accumulated_transforms: List[Tuple[float, np.ndarray, np.ndarray]] = []  # 累积变换列表
        
        # 全局点云和轨迹
        self.global_points = []  # 全局点云
        self.global_colors = []  # 全局颜色
        self.trajectory = []     # 相机轨迹
        
        # 模型
        self.model = None
        self._load_model()
        
        # 可视化器
        self.viewer = None
        self._init_viewer(viewer_port)
    
    def _load_model(self):
        """加载DA3模型"""
        try:
            from depth_anything_3.api import DepthAnything3
            model_path = "/home/zhouyi/repo/Depth-Anything-3/checkpoints/DA3-BASE"
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
    
    def _init_viewer(self, port: int):
        """初始化可视化器"""
        try:
            from viewer import SLAMViewer
            self.viewer = SLAMViewer(port=port)
            print(f"Viewer initialized on port {port}")
        except ImportError as e:
            print(f"Failed to initialize viewer: {e}")
            self.viewer = None
    
    def load_image_paths(self, folder_path: str) -> List[str]:
        """加载图像路径列表"""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext), recursive=False))
        
        # 排序（按数字顺序）
        def extract_number(filename):
            base = os.path.basename(filename)
            name, _ = os.path.splitext(base)
            numbers = ''.join(filter(str.isdigit, name))
            return int(numbers) if numbers else 0
        
        image_paths.sort(key=extract_number)
        
        if not image_paths:
            print(f"Warning: No images found in {folder_path}")
            return []
        
        print(f"Found {len(image_paths)} images in {folder_path}")
        return image_paths
    
    def process_single_chunk(self, chunk_image_paths: List[str]) -> Dict:
        """
        处理单个块
        
        Args:
            chunk_image_paths: 块内图像路径列表
            
        Returns:
            预测结果字典
        """
        print(f"Processing chunk {self.chunk_count} with {len(chunk_image_paths)} images...")
        
        try:
            with torch.no_grad():
                torch.cuda.empty_cache()
                
                # 使用DA3处理整个块
                prediction = self.model.inference(
                    image=chunk_image_paths,
                    process_res=504,
                    process_res_method="upper_bound_resize",
                    export_dir=None,
                )
                
                # 整理预测结果
                result = {
                    'chunk_idx': self.chunk_count,
                    'image_paths': chunk_image_paths,
                    'processed_images': prediction.processed_images,  # [N, H, W, 3] uint8
                    'depth': np.squeeze(prediction.depth),           # [N, H, W] float32
                    'conf': prediction.conf,                         # [N, H, W] float32
                    'extrinsics': prediction.extrinsics,             # [N, 3, 4] float32 (w2c)
                    'intrinsics': prediction.intrinsics,             # [N, 3, 3] float32
                }
                
                return result
                
        except Exception as e:
            print(f"Error during DA3 chunk inference: {e}")
            raise
    
    def should_process_chunk(self) -> bool:
        """
        判断是否应该处理当前chunk
        
        Returns:
            是否应该处理chunk
        """
        # 第一chunk：需要收集足够的帧
        if self.chunk_count == 0:
            return len(self.frame_buffer) >= self.chunk_size
        
        # 后续chunk：需要足够的新帧
        return len(self.frame_buffer) >= self.chunk_size
    
    def prepare_chunk_data(self) -> List[str]:
        """
        从缓冲区准备chunk数据
        
        Returns:
            chunk_image_paths: chunk内的图像路径列表
        """
        # 确定要处理的帧
        if self.chunk_count == 0:
            # 第一chunk：取前chunk_size帧
            chunk_frames = list(self.frame_buffer)[:self.chunk_size]
        else:
            # 后续chunk：缓冲区中所有帧
            chunk_frames = list(self.frame_buffer)[:self.chunk_size]
        
        # 提取图像路径
        chunk_image_paths = [frame['path'] for frame in chunk_frames]
        
        return chunk_image_paths
    
    def update_buffer_after_chunk_processed(self):
        """处理完chunk后更新缓冲区"""
        if self.chunk_count == 0:
            # 第一chunk后保留最后overlap_size帧用于下一chunk
            if len(self.frame_buffer) > self.overlap_size:
                # 移除前chunk_size帧，保留最后overlap_size帧
                for _ in range(self.chunk_size - self.overlap_size):
                    if self.frame_buffer:
                        self.frame_buffer.popleft()
        else:
            # 后续chunk：保留最后overlap_size帧
            if len(self.frame_buffer) > self.overlap_size:
                # 移除前chunk_size帧，保留最后overlap_size帧
                for _ in range(self.chunk_size - self.overlap_size):
                    if self.frame_buffer:
                        self.frame_buffer.popleft()
    
    def process_chunk_alignment(self, prev_chunk_data: Dict, current_chunk_data: Dict) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        处理chunk对齐
        
        Args:
            prev_chunk_data: 前一个chunk的数据
            current_chunk_data: 当前chunk的数据
            
        Returns:
            s: 尺度因子
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
        """
        print(f"  Aligning chunk {self.chunk_count} with previous chunk...")
        
        # 提取重叠区域点云
        point_map1, point_map2 = extract_overlap_points(
            prev_chunk_data, current_chunk_data, self.overlap_size
        )
        
        # 提取重叠区域置信度
        conf1 = prev_chunk_data['conf'][-self.overlap_size:]
        conf2 = current_chunk_data['conf'][:self.overlap_size]
        
        # 对齐点云
        s, R, t = align_two_point_clouds(point_map1, point_map2, conf1, conf2)
        
        print(f"  Alignment successful")
        return s, R, t
    
    def merge_chunk_to_global(self, chunk_data: Dict, transform: Tuple[float, np.ndarray, np.ndarray]):
        """
        将chunk点云合并到全局点云
        
        Args:
            chunk_data: chunk数据
            transform: 变换 (s, R, t)
        """
        # 计算当前chunk的所有点云
        point_cloud = depth_to_point_cloud_vectorized(
            chunk_data['depth'],
            chunk_data['intrinsics'],
            chunk_data['extrinsics']
        )
        
        # 应用变换
        s, R, t = transform
        point_cloud_aligned = apply_sim3_transform(point_cloud, s, R, t)
        
        # 展平点云和颜色
        points_flat = point_cloud_aligned.reshape(-1, 3)
        colors_flat = chunk_data['processed_images'].reshape(-1, 3)
        confs_flat = chunk_data['conf'].reshape(-1)
        
        # 过滤并采样点云
        points_sampled, colors_sampled = filter_point_cloud(
            points_flat, colors_flat, confs_flat,
            min_depth=0.1, max_depth=50.0, 
            min_confidence=self.min_confidence, sample_ratio=0.2
        )
        
        if len(points_sampled) > 0:
            # 添加到全局点云
            self.global_points.append(points_sampled)
            self.global_colors.append(colors_sampled)
            
            print(f"  Added {len(points_sampled)} points from chunk {chunk_data['chunk_idx']}")
    
    def update_trajectory(self, chunk_data: Dict, transform: Tuple[float, np.ndarray, np.ndarray]):
        """更新相机轨迹"""
        N = len(chunk_data['extrinsics'])
        
        for i in range(N):
            # 应用变换到相机位姿
            extrinsic_global = transform_camera_pose(chunk_data['extrinsics'][i], *transform)
            
            # 提取相机位置（从c2w矩阵）
            w2c_global = np.eye(4)
            w2c_global[:3, :4] = extrinsic_global
            c2w_global = np.linalg.inv(w2c_global)
            position = c2w_global[:3, 3]
            
            self.trajectory.append(position)
    
    def update_viewer(self, chunk_data: Dict, transform: Tuple[float, np.ndarray, np.ndarray]):
        """更新可视化器"""
        if self.viewer is None:
            return
        
        N = len(chunk_data['extrinsics'])
        
        for i in range(N):
            # 计算帧索引
            frame_idx = self.frame_count - len(self.frame_buffer) - (self.chunk_size - N) + i
            
            # 提取原始图像（需要转换为CHW格式）
            image_hwc = chunk_data['processed_images'][i]  # [H, W, 3]
            image_chw = image_hwc.transpose(2, 0, 1) / 255.0  # [3, H, W] 归一化
            
            # 获取深度图
            depth = chunk_data['depth'][i]  # [H, W]
            
            # 获取置信度图
            conf = chunk_data['conf'][i]  # [H, W]
            
            # 获取变换后的外参
            extrinsic_global = transform_camera_pose(chunk_data['extrinsics'][i], *transform)
            
            # 获取内参
            intrinsic = chunk_data['intrinsics'][i]  # [3, 3]
            
            # 添加到可视化器
            self.viewer.add_keyframe(
                image=image_chw,
                depth=depth,
                conf=conf,
                extrinsic=extrinsic_global,
                intrinsic=intrinsic,
                frame_idx=frame_idx
            )
        
        print(f"  Chunk {chunk_data['chunk_idx']} visualized ({N} frames)")
    
    def process_frame(self, image_path: str):
        """
        处理一帧图像（流式处理）
        
        Args:
            image_path: 图像路径
        """
        self.frame_count += 1
        print(f"\nProcessing frame {self.frame_count}: {os.path.basename(image_path)}")
        
        # 存储图像路径到缓冲区
        frame_data = {
            'idx': self.frame_count,
            'path': image_path,
        }
        self.frame_buffer.append(frame_data)
        print(f"  Added to buffer. Buffer size: {len(self.frame_buffer)}")
        
        # 检查是否需要处理chunk
        if self.should_process_chunk():
            print(f"\n  Processing chunk {self.chunk_count}...")
            
            # 1. 准备chunk数据
            chunk_image_paths = self.prepare_chunk_data()
            
            # 2. 处理chunk
            chunk_data = self.process_single_chunk(chunk_image_paths)
            
            # 3. 添加到chunk列表
            self.chunk_data_list.append(chunk_data)
            
            # 4. 对齐处理（如果不是第一个chunk）
            if self.chunk_count > 0:
                prev_chunk_data = self.chunk_data_list[self.chunk_count - 1]
                s, R, t = self.process_chunk_alignment(prev_chunk_data, chunk_data)
                self.sim3_transforms.append((s, R, t))
            
            # 5. 更新累积变换
            self.accumulated_transforms = accumulate_sim3_transforms(self.sim3_transforms)
            
            # 6. 获取当前chunk的变换
            if self.chunk_count == 0:
                transform = (1.0, np.eye(3), np.zeros(3))
            else:
                transform = self.accumulated_transforms[self.chunk_count]
            
            # 7. 合并到全局点云
            self.merge_chunk_to_global(chunk_data, transform)
            
            # 8. 更新轨迹
            self.update_trajectory(chunk_data, transform)
            
            # 9. 更新可视化
            self.update_viewer(chunk_data, transform)
            
            # 10. 更新缓冲区
            self.update_buffer_after_chunk_processed()
            
            # 11. 增加chunk计数
            self.chunk_count += 1
            
            print(f"  Chunk {chunk_data['chunk_idx']} processed successfully")
            
            # 等待一会儿以便观察
            time.sleep(3)
    
    def run_slam(self, folder_path: str):
        """
        运行完整的SLAM流程（流式处理）
        
        Args:
            folder_path: 图像文件夹路径
        """
        print("=" * 50)
        print("Starting DA3-Streaming SLAM (Streaming Mode)...")
        print(f"Chunk size: {self.chunk_size}, overlap_size: {self.overlap_size}")
        print("=" * 50)
        
        # 加载所有图像路径
        image_paths = self.load_image_paths(folder_path)
        if not image_paths:
            print(f"Warning: No images found in {folder_path}")
            return
        
        # 流式处理每一帧
        try:
            for i, img_path in enumerate(image_paths):
                self.process_frame(img_path)
                
                # 显示进度
                if (i + 1) % 10 == 0:
                    print(f"\nProgress: {i+1}/{len(image_paths)} frames, "
                          f"{self.chunk_count} chunks processed")
        
        except KeyboardInterrupt:
            print("\nSLAM interrupted by user")
        except Exception as e:
            print(f"Error during SLAM processing: {e}")
            import traceback
            traceback.print_exc()
        
        # 完成处理
        print("\n" + "=" * 50)
        print("SLAM process completed!")
        print(f"Total frames: {self.frame_count}")
        print(f"Total chunks: {self.chunk_count}")
        print(f"Trajectory length: {len(self.trajectory)}")
        if self.global_points:
            total_points = sum(len(p) for p in self.global_points)
            print(f"Total global points: {total_points}")
        print("=" * 50)