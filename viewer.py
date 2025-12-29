# viewer.py
import numpy as np
import viser
import viser.transforms as viser_tf
import threading
import time
import cv2
from typing import List, Optional
from tqdm.auto import tqdm
from src.vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map

class SLAMViewer:
    def __init__(self, port: int = 8080, vis_stride: int = 1, vis_point_size: float = 0.003):
        """
        初始化SLAM可视化器
        
        Args:
            port: 网页服务器端口
            vis_stride: 可视化点云采样间隔
            vis_point_size: 可视化点大小
        """
        self.port = port
        self.vis_stride = vis_stride
        self.vis_point_size = vis_point_size
        
        # 可视化状态
        self.server = None
        self.point_cloud = None
        self.frames: List[viser.FrameHandle] = []
        self.frustums: List[viser.CameraFrustumHandle] = []
        
        # 数据存储
        self.all_points = []
        self.all_colors = []
        self.all_confidences = []
        self.camera_poses = []
        self.frame_indices = []
        self.total_points = 0
        
        # GUI控件
        self.gui_show_frames = None
        self.gui_points_conf = None
        self.gui_frame_selector = None
        
        # 启动服务器
        self._start_server()
        
    def _start_server(self):
        """启动viser服务器"""
        print(f"Starting SLAM viewer on port {self.port}")
        self.server = viser.ViserServer(host="0.0.0.0", port=self.port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        
        # 初始化GUI控件
        self._setup_gui()
        
        # 初始化点云
        self.point_cloud = self.server.scene.add_point_cloud(
            name="slam_pcd",
            points=np.array([]).reshape(-1, 3),
            colors=np.array([]).reshape(-1, 3),
            point_size=self.vis_point_size,
            point_shape="circle",
        )
        
    def _setup_gui(self):
        """设置GUI控件"""
        self.gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.gui_points_conf = self.server.gui.add_slider(
            "Confidence Threshold", min=0, max=100, step=0.1, initial_value=50.0
        )
        self.gui_frame_selector = self.server.gui.add_dropdown(
            "Show Points from Frames", options=["All"], initial_value="All"
        )
        
        # 设置回调
        @self.gui_points_conf.on_update
        def _(_) -> None:
            self._update_point_cloud()
            
        @self.gui_frame_selector.on_update
        def _(_) -> None:
            self._update_point_cloud()
            
        @self.gui_show_frames.on_update
        def _(_) -> None:
            self._toggle_camera_visibility()
    
    def add_keyframe(self, 
                    image: np.ndarray, 
                    depth: np.ndarray,
                    conf: np.ndarray,
                    extrinsic: np.ndarray,
                    intrinsic: np.ndarray,
                    frame_idx: int):
        """
        添加关键帧到可视化器
        
        Args:
            image: 图像 (3, H, W) 或 (H, W, 3)，归一化到[0, 1]
            depth: 深度图 (H, W) 或 (H, W, 1)
            conf: 置信度图 (H, W)
            extrinsic: 外参矩阵 (3, 4) w2c
            intrinsic: 内参矩阵 (3, 3)
            frame_idx: 帧索引
        """
        # 1. 确保深度图为正确的形状
        if depth.ndim == 2:
            depth = depth[:, :, np.newaxis]
        
        H_dep, W_dep, _ = depth.shape
        
        # 2. 处理图像颜色
        if image.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
            H_img, W_img = image.shape[1], image.shape[2]
            colors = image.transpose(1, 2, 0)
        else:
            H_img, W_img = image.shape[0], image.shape[1]
            colors = image
        
        # 3. 调整图像大小以匹配深度图尺寸
        if (H_img, W_img) != (H_dep, W_dep):
            colors = cv2.resize(colors, (W_dep, H_dep), interpolation=cv2.INTER_LINEAR)
        
        colors_uint8 = (colors * 255).astype(np.uint8)
        
        # 4. 反投影深度图得到3D点
        world_points = unproject_depth_map_to_point_map(
            depth[np.newaxis, ...],  # 添加batch维度
            extrinsic[np.newaxis, ...],
            intrinsic[np.newaxis, ...]
        )[0]  # 移除batch维度，得到(H_dep, W_dep, 3)
        
        # 5. 应用步长采样 - 确保掩码与所有数组维度匹配
        mask = np.zeros((H_dep, W_dep), dtype=bool)
        mask[::self.vis_stride, ::self.vis_stride] = True
        
        # 6. 应用掩码并展平数据
        points_masked = world_points[mask]
        colors_masked = colors_uint8[mask]
        conf_masked = conf[mask] if conf.shape == (H_dep, W_dep) else np.ones(len(points_masked))
        
        # 7. 过滤掉无效点（深度为0或NaN的点）
        valid_mask = (
            (points_masked[:, 2] > 0.1) &  # 深度大于0.1
            (points_masked[:, 2] < 50.0) & # 深度小于50米
            np.all(np.isfinite(points_masked), axis=1)  # 所有坐标都是有限值
        )
        
        if np.any(valid_mask):
            points_valid = points_masked[valid_mask]
            colors_valid = colors_masked[valid_mask]
            conf_valid = conf_masked[valid_mask] if conf_masked.ndim > 0 else np.ones(len(points_valid))
            
            # 8. 存储数据
            self.all_points.append(points_valid)
            self.all_colors.append(colors_valid)
            self.all_confidences.append(conf_valid)
            self.camera_poses.append(extrinsic)
            
            # 记录帧索引
            num_points = len(points_valid)
            self.frame_indices.extend([frame_idx] * num_points)
            self.total_points += num_points
            
            print(f"  Added {num_points} points from frame {frame_idx}, total points: {self.total_points}")
        
        # 9. 更新帧选择器
        options = ["All"] + [str(i) for i in range(frame_idx + 1)]
        if self.gui_frame_selector.options != options:
            self.gui_frame_selector.options = options
        
        # 10. 添加相机位姿可视化
        self._add_camera_visualization(extrinsic, colors, frame_idx)
        
        # 11. 更新点云
        self._update_point_cloud()
        
    def _add_camera_visualization(self, extrinsic: np.ndarray, image: np.ndarray, frame_idx: int):
        """添加相机可视化（坐标轴和视锥体）"""
        # 将w2c转换为c2w
        cam_to_world = closed_form_inverse_se3(extrinsic[np.newaxis, ...])[0]  # (4, 4)
        
        # 创建SE3变换
        T_world_camera = viser_tf.SE3.from_matrix(cam_to_world)
        
        # 添加坐标轴
        frame = self.server.scene.add_frame(
            f"frame_{frame_idx}",
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            axes_length=0.1,
            axes_radius=0.003,
            origin_radius=0.005,
            visible=True
        )
        self.frames.append(frame)
        
        # 添加视锥体
        if len(image.shape) == 3 and image.shape[2] == 3:  # (H, W, 3)
            img_display = (image * 255).astype(np.uint8)
        elif len(image.shape) == 3 and image.shape[0] == 3:  # (3, H, W)
            img_display = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
        else:
            img_display = (image * 255).astype(np.uint8)
            
        h, w = img_display.shape[:2]
        
        # 计算FOV
        fx = 1.1 * h
        fov = 2 * np.arctan2(h / 2, fx)
        
        frustum = self.server.scene.add_camera_frustum(
            f"frame_{frame_idx}/frustum",
            fov=fov,
            aspect=w / h,
            scale=0.08,
            image=img_display,
            line_width=1.5,
            visible=True
        )
        self.frustums.append(frustum)
        
        # 添加点击回调
        @frustum.on_click
        def _(_) -> None:
            for client in self.server.get_clients().values():
                client.camera.wxyz = frame.wxyz
                client.camera.position = frame.position
    
    def _update_point_cloud(self):
        """根据GUI设置更新点云"""
        if len(self.all_points) == 0:
            return
            
        # 合并所有点
        all_points_combined = np.vstack(self.all_points) if len(self.all_points) > 0 else np.array([]).reshape(-1, 3)
        all_colors_combined = np.vstack(self.all_colors) if len(self.all_colors) > 0 else np.array([]).reshape(-1, 3)
        
        if len(all_points_combined) == 0:
            return
            
        # 合并置信度
        all_conf_combined = np.hstack(self.all_confidences) if len(self.all_confidences) > 0 else np.ones(len(all_points_combined))
        
        # 应用置信度阈值
        if len(all_conf_combined) > 0 and np.any(all_conf_combined > 0):
            threshold_val = np.percentile(all_conf_combined[all_conf_combined > 0], 
                                         min(self.gui_points_conf.value, 99.9))
            conf_mask = (all_conf_combined >= threshold_val)
        else:
            conf_mask = np.ones(len(all_points_combined), dtype=bool)
        
        # 应用帧选择器
        if self.gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            try:
                selected_idx = int(self.gui_frame_selector.value)
                frame_mask = np.array(self.frame_indices) == selected_idx
            except:
                frame_mask = np.ones_like(conf_mask, dtype=bool)
                
        combined_mask = conf_mask & frame_mask
        
        # 更新点云
        if np.any(combined_mask) and len(all_points_combined) > 0:
            self.point_cloud.points = all_points_combined[combined_mask]
            self.point_cloud.colors = all_colors_combined[combined_mask]
            print(f"  Updated point cloud: {np.sum(combined_mask)} points displayed")
    
    def _toggle_camera_visibility(self):
        """切换相机可视化可见性"""
        visible = self.gui_show_frames.value
        for frame in self.frames:
            frame.visible = visible
        for frustum in self.frustums:
            frustum.visible = visible
    
    def clear(self):
        """清除所有可视化数据"""
        # 清除点云
        if self.point_cloud is not None:
            self.point_cloud.points = np.array([]).reshape(-1, 3)
            self.point_cloud.colors = np.array([]).reshape(-1, 3)
        
        # 清除相机可视化
        for frame in self.frames:
            frame.remove()
        self.frames.clear()
        
        for frustum in self.frustums:
            frustum.remove()
        self.frustums.clear()
        
        # 清除数据
        self.all_points.clear()
        self.all_colors.clear()
        self.all_confidences.clear()
        self.camera_poses.clear()
        self.frame_indices.clear()
        self.total_points = 0
        
        # 重置GUI
        self.gui_frame_selector.options = ["All"]
        print("Viewer cleared")
    
    def run(self, background: bool = False):
        """运行服务器"""
        if background:
            def server_loop():
                while True:
                    time.sleep(0.001)
            thread = threading.Thread(target=server_loop, daemon=True)
            thread.start()
            print("Viewer running in background thread")
        else:
            print("Viewer running in main thread. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(0.01)
            except KeyboardInterrupt:
                print("\nViewer stopped by user")