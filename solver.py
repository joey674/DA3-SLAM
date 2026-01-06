# solver.py
import numpy as np
import torch
import time
from typing import List, Dict, Tuple
from collections import deque

from align import (
    extract_overlap_chunk_prediction,
    align_two_point_clouds,
)

from geometry import (
    accumulate_sim3_transforms,
    transform_camara_extrinsics,
)

from utils import (
    load_image,
    extract_keyframe,                   
)


class SLAMSolver:
    def __init__(self, 
                 image_dir,
                 config):
        """
        初始化SLAM求解器
        
        Args:
        
        """
        self.config = config
        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap_size = self.config["Model"]["overlap_size"]
        self.image_dir = image_dir
        
        # 帧计数器
        self.frame_count = 0
        self.chunk_count = 0
        
        # 数据存储 - 流式处理
        self.frame_buffer: deque = deque(maxlen=self.chunk_size * 2)  # 帧缓冲区
        self.chunk_prediction_list: List[Dict] = []  # 已处理chunk数据
        self.sim3_transforms: List[Tuple[float, np.ndarray, np.ndarray]] = []  # SIM(3)变换列表
        self.accumulated_transforms: List[Tuple[float, np.ndarray, np.ndarray]] = []  # 累积变换列表
        
        # 模型
        self.model = None
        self.load_model()
        
        # 可视化器
        self.viewer = None
        self.init_viewer()
    
    def load_model(self):
        """加载DA3模型"""
        # 设备设置
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
        if self.chunk_count == 0:
            # 第一chunk后保留最后overlap_size帧用于下一chunk
            if len(self.frame_buffer) > self.overlap_size:
                # 移除前chunk_size帧 保留最后overlap_size帧
                for _ in range(self.chunk_size - self.overlap_size):
                    if self.frame_buffer:
                        self.frame_buffer.popleft()
        else:
            # 后续chunk  保留最后overlap_size帧
            if len(self.frame_buffer) > self.overlap_size:
                # 移除前chunk_size帧 保留最后overlap_size帧
                for _ in range(self.chunk_size - self.overlap_size):
                    if self.frame_buffer:
                        self.frame_buffer.popleft()
    
    def update_viewer(self, chunk_data: Dict, transform: Tuple[float, np.ndarray, np.ndarray]):
        """
        更新可视化器
        Args:
            chunk_data: chunk的数据
            transform: 这个chunk相对于全局坐标系(也就是第一个chunk)的变换
            
        Returns:
            s: 尺度因子
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
        """
        if self.viewer is None:
            return
        
        N = len(chunk_data['extrinsics'])
        
        frame_indices = list(range(0, N))
        
        for i in frame_indices:
            # 计算帧索引
            frame_idx = self.frame_count - len(self.frame_buffer) - (self.chunk_size - N) + i
            
            # 提取原始图像 需要转换为CHW格式
            image_hwc = chunk_data['processed_images'][i]  # [H, W, 3]
            image_chw = image_hwc.transpose(2, 0, 1) / 255.0  # [3, H, W] 归一化
            
            # 获取深度图
            depth = chunk_data['depth'][i]  # [H, W]
            print(f"{depth.shape}")
            
            # 获取置信度图
            conf = chunk_data['conf'][i]  # [H, W]
            
            # 获取变换后的外参
            extrinsic_global = transform_camara_extrinsics(chunk_data['extrinsics'][i], *transform)
            
            # 获取内参
            intrinsic = chunk_data['intrinsics'][i]  # [3, 3]
            
            # 添加到可视化器
            self.viewer.add_frame(
                image=image_chw,
                depth=depth,
                conf=conf,
                extrinsic=extrinsic_global,
                intrinsic=intrinsic,
                frame_idx=frame_idx
            )
    
    def process_chunk_alignment(self, prev_chunk_prediction: Dict, cur_chunk_prediction: Dict) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        处理chunk对齐
        
        Args:
            prev_chunk_prediction: 前一个chunk的数据
            cur_chunk_prediction: 当前chunk的数据
            
        Returns:
            s: 尺度因子
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3]
        """
        print(f"  Aligning chunk {self.chunk_count} with previous chunk...")
        
        # 提取重叠区域点云和置信度
        point_map1, point_map2, conf1, conf2 = extract_overlap_chunk_prediction(
            prev_chunk_prediction, cur_chunk_prediction, self.overlap_size
        )
        
        # 对齐点云
        s, R, t = align_two_point_clouds(point_map1, point_map2, conf1, conf2)

        return s, R, t
    
    def run_single_chunk_prediction(self, chunk_image_paths: List[str]) -> Dict:
        """
        处理单个块
        
        Args:
            chunk_image_paths: 块内图像路径列表
            
        Returns:
            预测结果字典
        """
        print(f"  Predict single chunk {self.chunk_count} with {len(chunk_image_paths)} images through da3...")
        

        torch.cuda.empty_cache()
        with torch.no_grad():
            
            prediction = self.model.inference(
                    image=chunk_image_paths,
                    process_res_method="upper_bound_resize",
                )
            
            depth = prediction.depth
            # depth = np.squeeze(prediction.depth)
            # print("prediction")
            # print(f"{prediction.depth.shape}")
            # print(f"{depth.shape}")
            
            # 整理预测结果
            result = {
                'chunk_idx': self.chunk_count,
                'image_paths': chunk_image_paths,
                'processed_images': prediction.processed_images, # [N, H, W, 3] uint8
                'depth': depth,           # [N, H, W] float32
                'conf': prediction.conf,                         # [N, H, W] float32
                'extrinsics': prediction.extrinsics,             # [N, 3, 4] float32 (w2c)
                'intrinsics': prediction.intrinsics,             # [N, 3, 3] float32
            }
            
            
            return result
               
    def load_chunk_image_paths(self) -> List[str]:
        """
        从缓冲区准备chunk数据
        
        Returns:
            chunk_image_paths: chunk内的图像路径列表
        """
        # 确定要处理的帧
        if self.chunk_count == 0:
            # 第一chunk  取前chunk_size帧
            chunk_frames = list(self.frame_buffer)[:self.chunk_size]
        else:
            # 后续chunk  缓冲区中所有帧
            chunk_frames = list(self.frame_buffer)[:self.chunk_size]
        
        # 提取图像路径
        chunk_image_paths = [frame['path'] for frame in chunk_frames]
        
        return chunk_image_paths
    
    def should_run_chunk_prediction(self) -> bool:
        """
        判断是否应该处理当前chunk
        
        Returns:
            是否应该处理chunk
        """
        # 第一chunk  需要收集足够的帧
        if self.chunk_count == 0:
            return len(self.frame_buffer) >= self.chunk_size
        
        # 后续chunk  需要足够的新帧
        return len(self.frame_buffer) >= self.chunk_size    
    
    def process_frame(self, image_path: str):
        """
        处理一帧图像
        
        Args:
            image_path: 图像路径
        """
        self.frame_count += 1
        
        # 存储图像路径到frame_buffer
        frame_data = {
            'idx': self.frame_count,
            'path': image_path,
        }
        self.frame_buffer.append(frame_data)
        
        # 检查是否需要处理chunk
        if self.should_run_chunk_prediction():
            print(f"\n  Processing chunk {self.chunk_count}...")
            
            # 准备chunk数据
            chunk_image_paths = self.load_chunk_image_paths()
            
            # 处理chunk
            cur_chunk_prediction = self.run_single_chunk_prediction(chunk_image_paths)
            
            # 添加到chunk列表
            self.chunk_prediction_list.append(cur_chunk_prediction)
            
            # 对齐处理 (如果不是第一个chunk )
            if self.chunk_count > 0:
                pre_chunk_prediction = self.chunk_prediction_list[self.chunk_count - 1]
                s, R, t = self.process_chunk_alignment(pre_chunk_prediction, cur_chunk_prediction)
                self.sim3_transforms.append((s, R, t))
            
            # 更新累积变换
            self.accumulated_transforms = accumulate_sim3_transforms(self.sim3_transforms)
            
            # 获取当前chunk的变换(相对于第一个chunk的变换)
            if self.chunk_count == 0:
                transform = (1.0, np.eye(3), np.zeros(3))
            else:
                transform = self.accumulated_transforms[self.chunk_count]
            
            # 更新可视化
            self.update_viewer(cur_chunk_prediction, transform)
            
            # 更新缓冲区
            self.update_buffer_after_chunk_processed()
            
            # 增加chunk计数
            self.chunk_count += 1
            
            print(f"  Chunk {cur_chunk_prediction['chunk_idx']} processed successfully")
            
            # 等待一会
            time.sleep(self.config["Model"]["sleep_demo"])
            print("  Sleep for observation")
            print("#"*50)
   
   

    def run(self):
        """
        运行完整的SLAM流程 流式处理
        
        Args:
            folder_path: 图像文件夹路径
        """
        print("=" * 50)
        print("Starting DA3-SLAM ...")
        print("=" * 50)
        
        image_dir = self.image_dir
        
        # 加载所有图像路径
        image_paths = load_image(image_dir)
        if not image_paths:
            print(f"Warning: No images found in {image_dir}")
            return
        
        # 抽取关键帧
        image_paths = extract_keyframe(image_paths,self.config["Model"]["keyframe_interval"])
        
        # 流式处理每一帧
        for i, img_path in enumerate(image_paths):
            self.process_frame(img_path)
        
        # 完成处理
        print("\n" + "=" * 50)
        print("SLAM process completed!")
        print("=" * 50)