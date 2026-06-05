import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Tuple, Dict, List
import cv2
import warnings
import sys
from sand_planner.utils.bspline import fit_trajectory_8cp, compute_bspline_arc_length_from_result
from sand_planner.utils.image import downscale_640x480_to_320x240, downscale_to_target_size
warnings.filterwarnings('ignore')


class SandPlannerBSplineDataset(Dataset):
    def __init__(self,
                 dataset_root: str,
                 sequence_length: int = 5,  # 默认：起始帧 + 之后 4 帧 / default: start frame + 4 forward frames
                 frame_skip: int = 3,  # 帧间跳帧间隔（0=连续，2=每 3 帧取 1） / skip interval between frames (0=consecutive, 2=every 3rd frame)
                 min_gap: int = 3,  # B-spline 拟合的最小间隔（至少需要 8 个点） / minimum gap for B-spline fitting (need at least 8 points)
                 max_gap: int = 25,  # 起止帧之间的最大间隔 / maximum gap between start and end frame
                 image_size: Tuple[int, int] = (480, 640),  # 原始深度图尺寸 (H, W) / original depth image size (H, W)
                 depth_scale: float = 1000.0,  # 由毫米转米（depth_pixel_value / 1000） / convert from mm to meters
                 max_depth: float = 8.0,  # 最大有效深度（米） / maximum effective depth in meters
                 normalize_depth: bool = True,
                 data_augmentation: bool = False,
                 min_trajectory_length: int = 5,
                 samples_per_epoch: int = 50000,
                 # 归一化相关 / normalization-related
                 normalize_targets: bool = False,
                 stats_json_path: str = "outputs/trajectory_stats.json",
                 norm_method: str = "percentile",  # 'percentile' 或/or 'zscore'
                 norm_margin: float = 0.10,
                 norm_clamp: bool = True,
                 # 深度图下采样开关：若为 True，先将 640x480 等比缩小到 320x240
                 # depth downscaling switch: if True, first shrink 640x480 to 320x240 proportionally
                 downscale_depth_half: bool = True,
                 # 静态序列数据增强配置 / static-sequence data-augmentation config
                 static_sequence_prob: float = 0.0,  # 生成静态序列的概率 / probability of generating a static sequence
                 static_sequence_strategy: str = "mixed",  # "full", "partial", "mixed"
                 static_frame_selection: str = "last",  # "first", "middle", "last", "random"
                 # 多场景支持 / multi-scene support
                 dataset_types: List[str] = None,  # 指定要加载的数据集类型，None 表示全部 / dataset types to load, None for all
                 scene_sampling_weights: Dict[str, float] = None,  # 不同场景的采样权重 / per-scene sampling weights
                 # 数据集划分支持 / dataset split support
                 split_mode: str = "all",  # "all", "train", "val"
                 train_ratio: float = 0.9,  # 训练集比例 / training-split ratio
                 random_seed: int = 42,  # 用于稳定划分的随机种子 / random seed for stable splitting
                 # 轨迹插值方法 / trajectory interpolation method
                 trajectory_interpolation: str = 'cubic_spline',  # 'bspline' 或/or 'cubic_spline'
                 # 预测模式 / prediction mode
                 prediction_mode: str = 'control_points',  # 'control_points' 或/or 'waypoints'
                 # B-spline 控制点数量（训练/推理必须一致） / number of B-spline control points (must match between training and inference)
                 num_control_points: int = 8,
                 # 弧长上限（米），超过此长度的轨迹会被截断后重新拟合；None 表示不限制
                 # arc-length upper bound (meters); trajectories longer than this are truncated then re-fitted; None means no limit
                 max_arc_length: float = None):
        """
        输出 B-spline 控制点的 SanD-planner 数据集（支持多场景）。 / SanD-planner dataset with B-spline control points output (multi-scene support).

        Args:
            dataset_root: 包含多个场景目录的数据集根路径 / path to dataset folder containing multiple scene directories
            sequence_length: 深度图数量（历史帧 + 当前帧） / number of depth images (historical frames + current frame)
            frame_skip: 历史帧之间的跳帧间隔（0=连续，3=每 4 帧回取一帧） / skip interval between historical frames (0=consecutive, 3=every 4th frame back in time)
            min_gap: 起止帧之间的最小间隔（end >= start + min_gap） / minimum gap between start and end frame
            max_gap: 起止帧之间的最大间隔（end <= start + max_gap） / maximum gap between start and end frame
            image_size: 深度图目标尺寸 (H, W) / target size for depth images (H, W)
            depth_scale: 深度值转换缩放因子（如 1000 表示 mm->m） / scale factor to convert depth values (e.g. 1000 for mm->m)
            max_depth: 用于截断的最大深度值 / maximum depth value for clipping
            normalize_depth: 是否将深度归一化到 [0, 1] / whether to normalize depth to [0, 1]
            data_augmentation: 是否应用随机数据增强 / whether to apply random augmentations
            min_trajectory_length: B-spline 拟合所需的最小轨迹长度 / minimum trajectory length for B-spline fitting
            samples_per_epoch: 每个 epoch 抽取的样本数（__len__） / number of samples to draw per epoch (__len__)
            dataset_types: 要加载的数据集类型列表（如 ['indoor', 'stair']），None 表示全部 / list of dataset types to load (e.g. ['indoor', 'stair']), None for all
            scene_sampling_weights: 不同场景的采样权重，None 表示均匀采样 / weights for sampling different scenes, None for uniform
            split_mode: 数据集划分模式 —— "all": 使用全部数据，"train": 训练集，"val": 验证集 / dataset split mode - "all": all data, "train": training split, "val": validation split
            train_ratio: 训练/验证划分比例（如 0.9 = 90% 训练，10% 验证） / ratio for train/val split (e.g. 0.9 = 90% train, 10% val)
            random_seed: 用于稳定训练/验证划分的随机种子 / random seed for stable train/val splitting

        Note:
            - 支持多个场景数据集：dataset_indoor、dataset_stair、dataset_campus 等。 / supports multiple scene datasets: dataset_indoor, dataset_stair, dataset_campus, etc.
            - 深度序列包含截至当前帧的历史帧。 / depth sequence includes historical frames leading up to current frame.
            - 历史采样：[过去帧, ..., 当前帧]，跳帧间隔可配置。 / historical sampling: [past_frame, ..., current_frame] with configurable skip interval.
            - 轨迹用 B-spline 拟合并输出为 8 个控制点 (8, 3)。 / trajectory is fitted with B-spline and output as 8 control points (8, 3).
            - 无需 padding —— 始终返回恰好 8 个控制点。 / no padding needed - always returns exactly 8 control points.
        """
        self.dataset_root = dataset_root
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.min_gap = max(min_gap, min_trajectory_length)  # 确保有足够的点用于 B-spline / ensure enough points for B-spline
        self.max_gap = max_gap
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.normalize_depth = normalize_depth
        self.data_augmentation = data_augmentation
        self.min_trajectory_length = min_trajectory_length
        # 每个 epoch 的样本数 / number of samples per epoch
        self.samples_per_epoch = int(samples_per_epoch)

        # 归一化配置 / normalization config
        self.normalize_targets = normalize_targets
        self.stats_json_path = stats_json_path
        self.norm_method = norm_method
        self.norm_margin = float(norm_margin)
        self.norm_clamp = bool(norm_clamp)
        self._normalizer = None
        # 深度图是否先进行 1/2 下采样（640x480 -> 320x240）
        # whether to first downsample the depth image by 1/2 (640x480 -> 320x240)
        self.downscale_depth_half = bool(downscale_depth_half)

        # 轨迹插值方法 / trajectory interpolation method
        self.trajectory_interpolation = trajectory_interpolation
        self.prediction_mode = prediction_mode
        # B-spline 控制点数量（训练/推理必须一致） / number of B-spline control points (must match between training and inference)
        self.num_control_points = int(num_control_points)
        self.max_arc_length = max_arc_length

        if self.max_arc_length is not None:
            print(f"🔧 弧长上限: {self.max_arc_length}m（超过将截断轨迹后重新拟合）")

        if self.prediction_mode == 'waypoints':
            print("🔧 预测模式: Waypoints (固定0.2m间隔采样)")
        else:
            if self.trajectory_interpolation == 'cubic_spline':
                from sand_planner.utils.traj_opt import TrajOpt
                self.traj_opt = TrajOpt()
                print("🔧 数据加载器使用 Cubic Spline 拟合控制点")
            else:
                print("🔧 数据加载器使用 B-Spline 拟合控制点")
        
        # 静态序列数据增强配置 / static-sequence data-augmentation config
        self.static_sequence_prob = max(0.0, min(1.0, static_sequence_prob))  # 限制在 [0, 1] / clamp to [0, 1]
        self.static_sequence_strategy = static_sequence_strategy
        self.static_frame_selection = static_frame_selection

        # 统计计数器 / statistics counters
        self.static_sequence_count = 0
        self.total_sequence_count = 0

        # 多场景支持 / multi-scene support
        self.dataset_types = dataset_types
        self.scene_sampling_weights = scene_sampling_weights or {}

        # 数据集划分支持 / dataset split support
        self.split_mode = split_mode
        self.train_ratio = train_ratio
        self.random_seed = random_seed

        # 发现所有可用的数据集 / discover all available datasets
        self.available_datasets = self._discover_datasets()

        # 过滤要使用的数据集 / filter the datasets to use
        if self.dataset_types is not None:
            self.available_datasets = {
                name: path for name, path in self.available_datasets.items() 
                if any(dt in name for dt in self.dataset_types)
            }

        print(f"发现数据集: {list(self.available_datasets.keys())}")

        # 缓存 run 信息以提高采样效率 / cache run info for efficient sampling
        self.run_info = {}  # 格式/format: {dataset_name: {run_id: run_data}}
        self.scene_runs = {}  # 格式/format: {dataset_name: [run_ids]}
        self._cache_all_run_info()

        # 应用数据集划分 / apply dataset split
        if self.split_mode in ["train", "val"]:
            self._apply_train_val_split()

        # 计算场景采样权重 / compute scene sampling weights
        self.scene_weights = self._compute_scene_weights()

        total_runs = sum(len(runs) for runs in self.scene_runs.values())
        print(f"总共加载了 {len(self.available_datasets)} 个场景，{total_runs} 个runs")
        for scene_name, runs in self.scene_runs.items():
            weight = self.scene_weights.get(scene_name, 0)
            print(f"  {scene_name}: {len(runs)} runs (权重: {weight:.3f})")
        
        print(f"Sequence length: {self.sequence_length} historical frames with skip interval: {self.frame_skip} (total history span: {(self.sequence_length-1)*(self.frame_skip+1)} frames)")
        print(f"B-spline fitting: min_gap={self.min_gap}, max_gap={self.max_gap}")
        print(f"Dataset samples_per_epoch = {self.samples_per_epoch}")
        print(f"Output: Fixed 8 control points (8, 3) - no padding needed!")
        if self.normalize_targets:
            print(f"Targets normalization enabled: method={self.norm_method}, margin={self.norm_margin*100:.0f}%, stats={self.stats_json_path}")
        if self.downscale_depth_half:
            target_h, target_w = self.image_size
            print(f"Depth downscale enabled: 640x480 -> {target_w}x{target_h} (one-step, nearest for depth)")
    
    def _discover_datasets(self) -> Dict[str, str]:
        """发现所有可用的数据集。 / Discover all available datasets."""
        datasets = {}

        if not os.path.exists(self.dataset_root):
            raise ValueError(f"Dataset root not found: {self.dataset_root}")

        # 检查 dataset_root 下的所有子目录 / inspect all subdirectories under dataset_root
        for item in os.listdir(self.dataset_root):
            item_path = os.path.join(self.dataset_root, item)
            if os.path.isdir(item_path) and item.startswith('dataset_'):
                # 检查是否包含 run_* 目录 / check whether it contains run_* directories
                run_dirs = [d for d in os.listdir(item_path)
                           if d.startswith('run_') and os.path.isdir(os.path.join(item_path, d))]
                if run_dirs:
                    datasets[item] = item_path
                    
        if not datasets:
            raise ValueError(f"No valid datasets found in {self.dataset_root}")
            
        return datasets
    
    def _cache_all_run_info(self):
        """缓存所有数据集的 run 信息。 / Cache run info for all datasets."""
        for dataset_name, dataset_path in self.available_datasets.items():
            print(f"Loading dataset: {dataset_name}")

            # 获取该数据集的所有 run 目录 / collect all run directories of this dataset
            run_dirs = sorted([d for d in os.listdir(dataset_path)
                              if d.startswith('run_') and os.path.isdir(os.path.join(dataset_path, d))])
            
            dataset_run_info = {}
            valid_runs = []
            
            for run_dir in run_dirs:
                try:
                    run_path = os.path.join(dataset_path, run_dir)

                    # 加载轨迹文件 / load trajectory files
                    traj_xyz_path = os.path.join(run_path, 'traj_xyz.npy')
                    traj_yaw_path = os.path.join(run_path, 'traj_yaw.npy')
                    traj_pitch_path = os.path.join(run_path, 'traj_pitch.npy')
                    depth_dir = os.path.join(run_path, 'depth')
                    
                    if not all([os.path.exists(traj_xyz_path), 
                               os.path.exists(traj_yaw_path), 
                               os.path.exists(traj_pitch_path),
                               os.path.exists(depth_dir)]):
                        continue
                    
                    # 检查轨迹长度 / check trajectory lengths
                    traj_xyz = np.load(traj_xyz_path)
                    traj_yaw = np.load(traj_yaw_path)
                    traj_pitch = np.load(traj_pitch_path)

                    # 统计深度图数量 / count depth images
                    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
                    depth_count = len(depth_files)

                    # 校验一致性与最小长度 / validate consistency and minimum length
                    if (len(traj_xyz) == len(traj_yaw) == len(traj_pitch) == depth_count and
                        len(traj_xyz) >= self.min_trajectory_length):

                        # 为每个 run 附加所属数据集信息 / attach dataset info to each run
                        run_key = f"{dataset_name}_{run_dir}"
                        dataset_run_info[run_key] = {
                            'length': len(traj_xyz),
                            'traj_xyz': traj_xyz,
                            'traj_yaw': traj_yaw,
                            'traj_pitch': traj_pitch,
                            'dataset_name': dataset_name,
                            'dataset_path': dataset_path,
                            'run_dir': run_dir,
                            'run_path': run_path
                        }
                        valid_runs.append(run_key)
                    else:
                        if len(traj_xyz) < self.min_trajectory_length:
                            print(f"  Skipping {dataset_name}/{run_dir}: too short ({len(traj_xyz)} < {self.min_trajectory_length})")
                        else:
                            print(f"  Warning: Inconsistent data lengths in {dataset_name}/{run_dir}")
                        
                except Exception as e:
                    print(f"  Error processing {dataset_name}/{run_dir}: {e}")
                    continue
            
            # 存储该数据集的信息 / store this dataset's info
            self.run_info.update(dataset_run_info)
            self.scene_runs[dataset_name] = valid_runs
            
            print(f"  Loaded {len(valid_runs)} valid runs from {dataset_name}")
    
    def _apply_train_val_split(self):
        """应用训练/验证集划分。 / Apply train/validation split."""
        import hashlib
        import random

        print(f"应用数据集划分: split_mode={self.split_mode}, train_ratio={self.train_ratio}")

        # 收集所有 run_keys / collect all run_keys
        all_run_keys = []
        for scene_name, run_keys in self.scene_runs.items():
            all_run_keys.extend(run_keys)

        # 使用稳定的哈希方法进行划分 / split using a stable hashing scheme
        train_runs = []
        val_runs = []

        for run_key in all_run_keys:
            # 基于 run_key 和 random_seed 的哈希值进行稳定划分 / stable split based on hash of run_key + random_seed
            hash_input = f"{run_key}_{self.random_seed}"
            hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

            # 用哈希值的取模范围决定 train/val 归属 / use the modulo range of the hash to assign train/val
            if (hash_val % 10000) < (self.train_ratio * 10000):
                train_runs.append(run_key)
            else:
                val_runs.append(run_key)

        # 根据 split_mode 过滤 run_keys / filter run_keys by split_mode
        if self.split_mode == "train":
            selected_runs = set(train_runs)
            print(f"使用训练集: {len(train_runs)} runs")
        elif self.split_mode == "val":
            selected_runs = set(val_runs)
            print(f"使用验证集: {len(val_runs)} runs")
        else:
            selected_runs = set(all_run_keys)  # 理论上不应到达此分支 / should not reach here

        # 过滤 scene_runs / filter scene_runs
        filtered_scene_runs = {}
        filtered_run_info = {}

        for scene_name, run_keys in self.scene_runs.items():
            filtered_keys = [key for key in run_keys if key in selected_runs]
            if filtered_keys:  # 只保留非空的场景 / keep only non-empty scenes
                filtered_scene_runs[scene_name] = filtered_keys

                # 同时过滤 run_info / filter run_info accordingly
                for key in filtered_keys:
                    filtered_run_info[key] = self.run_info[key]

        # 更新实例变量 / update instance variables
        self.scene_runs = filtered_scene_runs
        self.run_info = filtered_run_info

        # 打印划分结果 / print split results
        total_selected = sum(len(runs) for runs in self.scene_runs.values())
        print(f"数据划分完成: 选择了 {total_selected} runs")
        for scene_name, runs in self.scene_runs.items():
            print(f"  {scene_name}: {len(runs)} runs")
    
    def _compute_scene_weights(self) -> Dict[str, float]:
        """计算场景采样权重。 / Compute scene sampling weights."""
        weights = {}
        total_runs = sum(len(runs) for runs in self.scene_runs.values())

        if not total_runs:
            return weights

        for scene_name, runs in self.scene_runs.items():
            if scene_name in self.scene_sampling_weights:
                # 使用用户指定的权重 / use the user-specified weight
                weights[scene_name] = self.scene_sampling_weights[scene_name]
            else:
                # 默认按 run 数量比例分配 / default: proportional to the number of runs
                weights[scene_name] = len(runs) / total_runs

        # 归一化权重 / normalize the weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _cache_run_info(self):
        """保持向后兼容的占位方法。 / Placeholder kept for backward compatibility."""
        # 此方法现已由 _cache_all_run_info 替代 / this method is now superseded by _cache_all_run_info
        pass

    def __len__(self):
        # 每个 epoch 返回固定数量的样本，便于控制训练步数
        # return a fixed number of samples per epoch to control the number of training steps
        return self.samples_per_epoch

    def _load_depth_image(self, run_key: str, frame_idx: int) -> torch.Tensor:
        """加载并预处理单张深度图。 / Load and preprocess a single depth image."""
        run_info = self.run_info[run_key]
        depth_path = os.path.join(run_info['run_path'], 'depth', f'depth_{frame_idx:04d}.png')

        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth image not found: {depth_path}")

        # 加载深度图（16 位 PNG） / load depth image (16-bit PNG)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise ValueError(f"Failed to load image: {depth_path}")

        # 转换为米并截断 / convert to meters and clip
        depth_array = depth_img.astype(np.float32) / self.depth_scale
        depth_array = np.clip(depth_array, 0, self.max_depth)

        # 可选：一步直接下采样到目标尺寸 / optional: directly downsample to target size in one step
        if self.downscale_depth_half:
            try:
                target_h, target_w = self.image_size
                # 直接使用通用下采样函数，一步到位 / use the generic downsampling function directly, in one step
                depth_array = downscale_to_target_size(depth_array, target_h, target_w, is_depth=True)
            except Exception:
                # 若失败则跳过该步骤，继续后续 resize / on failure, skip this step and fall through to the resize below
                pass

        # Resize 到期望尺寸 (H, W) —— 尺寸不匹配时才调整 / resize to the desired size (H, W) only if it does not match
        if depth_array.shape[:2] != self.image_size:
            # cv2.resize 接受 (width, height)，而 self.image_size 是 (height, width) / cv2.resize expects (width, height), but self.image_size is (height, width)
            target_height, target_width = self.image_size
            depth_array = cv2.resize(depth_array, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        # 按需归一化到 [0, 1] / normalize to [0, 1] if requested
        if self.normalize_depth:
            depth_array = depth_array / self.max_depth

        # 转为张量并添加通道维 / convert to tensor and add channel dimension
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)  # (1, H, W)

        return depth_tensor
    
    def _load_depth_sequence(self, run_key: str, start_idx: int, run_length: int) -> torch.Tensor:
        """
        加载以 start_idx 结尾的深度图序列。 / Load a sequence of depth images ending at start_idx.

        加载 5 帧：[start_idx-4, start_idx-3, start_idx-2, start_idx-1, start_idx]；
        若帧索引早于轨迹起点（frame_idx < 0），则使用第 0 帧。
        同时支持静态序列数据增强：以一定概率生成重复帧序列。

        Loads 5 frames: [start_idx-4, start_idx-3, start_idx-2, start_idx-1, start_idx];
        if before trajectory start (frame_idx < 0), use frame 0.
        Also supports static-sequence augmentation: with some probability generate a repeated-frame sequence.
        """
        depth_images = []

        # 加载历史序列（从过去到现在） / load the historical sequence (from past to present)
        for i in range(self.sequence_length):
            # 按跳帧间隔计算历史帧的帧索引 / compute the frame index of historical frames with the skip interval
            # 以 frame_skip=3、sequence_length=5、从 start_idx 起为例 / e.g. frame_skip=3, sequence_length=5, starting from start_idx:
            # i=0: frame_idx = start_idx - (5-1)*(3+1) = start_idx - 16  (最早的历史帧 / earliest historical frame)
            # i=1: frame_idx = start_idx - (4-1)*(3+1) = start_idx - 12
            # i=2: frame_idx = start_idx - (3-1)*(3+1) = start_idx - 8
            # i=3: frame_idx = start_idx - (2-1)*(3+1) = start_idx - 4
            # i=4: frame_idx = start_idx - (1-1)*(3+1) = start_idx - 0   (当前帧 / current frame)
            frame_idx = start_idx - (self.sequence_length - 1 - i) * (self.frame_skip + 1)

            # 若 frame_idx 为负，则使用第 0 帧（轨迹首帧） / if frame_idx is negative, use frame 0 (first frame of trajectory)
            if frame_idx < 0:
                frame_idx = 0

            depth_tensor = self._load_depth_image(run_key, frame_idx)
            depth_images.append(depth_tensor)

        depth_sequence = torch.stack(depth_images)  # 形状/shape: (sequence_length, 1, H, W)

        # 应用静态序列数据增强 / apply static-sequence augmentation
        if self.static_sequence_prob > 0:
            depth_sequence = self._apply_static_sequence_augmentation(depth_sequence)

        return depth_sequence
    
    def _apply_static_sequence_augmentation(self, depth_sequence: torch.Tensor) -> torch.Tensor:
        """
        应用静态序列数据增强。 / Apply static-sequence data augmentation.

        Args:
            depth_sequence: 原始深度序列 (sequence_length, 1, H, W) / original depth sequence (sequence_length, 1, H, W)

        Returns:
            增强后的深度序列。 / the augmented depth sequence.
        """
        import random

        self.total_sequence_count += 1

        # 根据概率决定是否应用静态增强 / decide whether to apply static augmentation based on probability
        if random.random() < self.static_sequence_prob:
            self.static_sequence_count += 1

            # 选择要重复的帧 / select the frame to repeat
            seq_len = depth_sequence.shape[0]

            if self.static_frame_selection == "first":
                selected_idx = 0
            elif self.static_frame_selection == "last":
                selected_idx = seq_len - 1
            elif self.static_frame_selection == "middle":
                selected_idx = seq_len // 2
            elif self.static_frame_selection == "random":
                selected_idx = random.randint(0, seq_len - 1)
            else:
                selected_idx = seq_len // 2  # 默认选择中间帧 / default to the middle frame

            selected_frame = depth_sequence[selected_idx]  # (1, H, W)

            # 应用不同的静态化策略 / apply different static strategies
            if self.static_sequence_strategy == "full":
                # 所有帧都使用选中的帧 / all frames use the selected frame
                static_sequence = selected_frame.unsqueeze(0).expand(seq_len, -1, -1, -1)
            elif self.static_sequence_strategy == "partial":
                # 随机选择部分帧进行静态化 / randomly pick a subset of frames to make static
                static_sequence = depth_sequence.clone()
                num_static = random.randint(2, seq_len - 1)
                static_indices = random.sample(range(seq_len), num_static)
                for idx in static_indices:
                    static_sequence[idx] = selected_frame
            elif self.static_sequence_strategy == "mixed":
                # 混合策略：随机选择上述策略 / mixed strategy: randomly choose one of the above
                if random.random() < 0.7:
                    # 70% 概率使用全静态 / 70% chance of fully static
                    static_sequence = selected_frame.unsqueeze(0).expand(seq_len, -1, -1, -1)
                else:
                    # 30% 概率使用部分静态 / 30% chance of partially static
                    static_sequence = depth_sequence.clone()
                    num_static = random.randint(2, seq_len - 1)
                    static_indices = random.sample(range(seq_len), num_static)
                    for idx in static_indices:
                        static_sequence[idx] = selected_frame
            else:
                # 默认使用全静态 / default to fully static
                static_sequence = selected_frame.unsqueeze(0).expand(seq_len, -1, -1, -1)

            return static_sequence

        return depth_sequence

    def get_static_sequence_stats(self) -> dict:
        """获取静态序列使用统计。 / Get static-sequence usage statistics."""
        if self.total_sequence_count == 0:
            return {"static_ratio": 0.0, "static_count": 0, "total_count": 0}
        
        return {
            "static_ratio": self.static_sequence_count / self.total_sequence_count,
            "static_count": self.static_sequence_count,
            "total_count": self.total_sequence_count
        }

    def _compute_relative_pose(self, xyz_start: np.ndarray, yaw_start: float, pitch_start: float,
                              xyz_end: np.ndarray, yaw_end: float, pitch_end: float) -> np.ndarray:
        """
        在起点坐标系下计算从起点到终点的相对位姿。 / Compute the relative pose from start to end in the start-frame coordinates.

        返回起点坐标系下的 [dx, dy, dz]（同时考虑 yaw 与 pitch 旋转）。
        变换顺序为：世界系 -> yaw 旋转 -> pitch 旋转 -> 局部系；
        因此其逆变换为：局部系 -> pitch^-1 -> yaw^-1 -> 世界系。

        Returns [dx, dy, dz] in the start frame (considering both yaw and pitch rotations).
        The transformation order is: World -> Yaw rotation -> Pitch rotation -> Local frame,
        so the inverse is: Local frame -> Pitch^-1 -> Yaw^-1 -> World.
        """
        # 世界坐标系下的平移差 / translation difference in world coordinates
        world_translation = xyz_end - xyz_start

        # 按相反顺序应用逆变换 / apply inverse transformations in reverse order:
        # 1. 逆 yaw 旋转（绕 Z 轴） / inverse yaw rotation (around Z-axis)
        # 2. 逆 pitch 旋转（绕 Y 轴） / inverse pitch rotation (around Y-axis)

        cos_yaw = np.cos(-yaw_start)
        sin_yaw = np.sin(-yaw_start)
        cos_pitch = np.cos(-pitch_start)
        sin_pitch = np.sin(-pitch_start)

        # 构造旋转：R_yaw_inv * R_pitch_inv * world_translation / build rotation: R_yaw_inv * R_pitch_inv * world_translation
        # 组合旋转 R_yaw_inv @ R_pitch_inv，等价于先做 pitch_inv 再做 yaw_inv
        # combined rotation R_yaw_inv @ R_pitch_inv is equivalent to applying pitch_inv first, then yaw_inv

        # 先应用逆 pitch 旋转（绕 Y 轴） / first apply inverse pitch rotation (around Y-axis)
        x_after_pitch = world_translation[0] * cos_pitch + world_translation[2] * sin_pitch
        y_after_pitch = world_translation[1]
        z_after_pitch = -world_translation[0] * sin_pitch + world_translation[2] * cos_pitch

        # 再应用逆 yaw 旋转（绕 Z 轴） / then apply inverse yaw rotation (around Z-axis)
        relative_x = x_after_pitch * cos_yaw - y_after_pitch * sin_yaw
        relative_y = x_after_pitch * sin_yaw + y_after_pitch * cos_yaw
        relative_z = z_after_pitch

        return np.array([relative_x, relative_y, relative_z], dtype=np.float32)
    
    def _compute_trajectory_relative_poses(self, traj_xyz: np.ndarray, traj_yaw: np.ndarray, 
                                         traj_pitch: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        """
        计算从起点到终点所有轨迹点的相对位姿。 / Compute relative poses for all trajectory points from start to end.

        返回 (trajectory_length, 3) 的数组，即相对于起点坐标系的 [dx, dy, dz]。
        Returns a (trajectory_length, 3) array of [dx, dy, dz] relative to the start frame.
        """
        start_xyz = traj_xyz[start_idx]
        start_yaw = traj_yaw[start_idx]
        start_pitch = traj_pitch[start_idx]

        # 取从起点到终点（含终点）的轨迹片段 / get the trajectory segment from start to end (inclusive)
        traj_segment_xyz = traj_xyz[start_idx:end_idx + 1]
        traj_segment_yaw = traj_yaw[start_idx:end_idx + 1]
        traj_segment_pitch = traj_pitch[start_idx:end_idx + 1]

        relative_poses = []
        for xyz, yaw, pitch in zip(traj_segment_xyz, traj_segment_yaw, traj_segment_pitch):
            rel_pose = self._compute_relative_pose(start_xyz, start_yaw, start_pitch, xyz, yaw, pitch)
            relative_poses.append(rel_pose)

        return np.array(relative_poses, dtype=np.float32)  # 形状/shape: (traj_len, 3)
    
    def _fit_bspline_control_points(self, trajectory_points: np.ndarray) -> tuple:
        """
        对轨迹拟合样条并返回控制点。 / Fit a spline to the trajectory and return control points.

        Args:
            trajectory_points: (N, 3) 的三维轨迹点数组 / (N, 3) array of 3D trajectory points

        Returns:
            tuple: (control_points, fitting_result)
                - control_points: (8, 3) 控制点数组 / (8, 3) array of control points
                - fitting_result: 包含拟合信息的字典 / dict with fitting information
        """
        try:
            # 根据 prediction_mode 和 trajectory_interpolation 选择方法 / choose the method based on prediction_mode and trajectory_interpolation
            if self.prediction_mode == 'waypoints':
                # Waypoints 模式：固定弧长间隔采样 / waypoints mode: sample at a fixed arc-length interval
                control_points, result = self._sample_waypoints(trajectory_points)
                return control_points, result
            elif self.trajectory_interpolation == 'cubic_spline':
                # 使用 Cubic Spline：优化 8 个控制点，最小化重建误差 / use cubic spline: optimize 8 control points to minimize reconstruction error
                # Cubic Spline 是插值样条，会严格经过这些控制点 / cubic spline is an interpolating spline, passing exactly through these control points
                control_points, result = self._fit_cubic_spline_control_points(trajectory_points)
                return control_points, result
            else:
                # 使用 B-spline 拟合 / use B-spline fitting
                result = fit_trajectory_8cp(trajectory_points, return_control_points=True,
                                            num_control_points=self.num_control_points)

            if result['success'] and 'control_points' in result:
                control_points = result['control_points']

                # 确保恰好有 num_control_points 个控制点 / ensure we have exactly num_control_points control points
                if len(control_points) != self.num_control_points:
                    # 数量不符时重采样到 num_control_points 个点 / if not exact, resample to num_control_points points
                    # 沿控制多边形做线性插值 / use linear interpolation along the control polygon
                    t_original = np.linspace(0, 1, len(control_points))
                    t_new = np.linspace(0, 1, self.num_control_points)

                    control_points_8 = np.zeros((self.num_control_points, 3))
                    for axis in range(3):
                        control_points_8[:, axis] = np.interp(t_new, t_original, control_points[:, axis])

                    control_points = control_points_8

                return control_points.astype(np.float32), result

            else:
                # B-spline 拟合失败，用线性插值生成 8 个控制点 / B-spline fitting failed, use linear interpolation to 8 control points
                print(f"B-spline fitting failed: {result.get('error_msg', 'unknown error')}")
                return self._linear_fallback_control_points(trajectory_points), result

        except Exception as e:
            print(f"B-spline fitting error: {e}")
            # 使用线性回退方案 / use the linear fallback
            dummy_result = {'success': False, 'error_msg': str(e)}
            return self._linear_fallback_control_points(trajectory_points), dummy_result
    
    def _fit_cubic_spline_control_points(self, trajectory_points: np.ndarray) -> tuple:
        """
        为 Cubic Spline 生成控制点。 / Generate control points for the cubic spline.

        策略：直接在 GT 轨迹上均匀采样 8 个点作为控制点。
        - 比优化方法快 100 倍以上。
        - 误差约 0.02m，对训练来说足够好。
        - Cubic Spline 会插值通过这些控制点。

        Strategy: uniformly sample 8 points along the GT trajectory as control points.
        - Over 100x faster than the optimization-based method.
        - Error is about 0.02m, good enough for training.
        - The cubic spline interpolates through these control points.

        Args:
            trajectory_points: (N, 3) GT 密集轨迹点（已转换为相对坐标，起点为原点） / (N, 3) GT dense trajectory points (in relative coordinates, start at origin)

        Returns:
            tuple: (control_points, fitting_result)
                - control_points: (8, 3) 包含原点的 8 个控制点 / (8, 3) eight control points including the origin
        """
        N = len(trajectory_points)
        m = self.num_control_points

        if N < m:
            # 点数不足，使用线性插值 / not enough points, use linear interpolation
            t_original = np.linspace(0, 1, N)
            t_new = np.linspace(0, 1, m)
            control_points = np.zeros((m, 3))
            for axis in range(3):
                control_points[:, axis] = np.interp(t_new, t_original, trajectory_points[:, axis])
            # 确保第一个点是原点 / ensure the first point is the origin
            control_points[0] = [0, 0, 0]
            result = {'success': True, 'method': 'cubic_spline_linear', 'num_points': N}
            return control_points.astype(np.float32), result

        # 均匀采样 m 个点作为控制点（含起点与终点） / uniformly sample m points as control points (including start and end)
        indices_8 = np.linspace(0, N-1, m, dtype=int)
        control_points = trajectory_points[indices_8].copy()

        # 确保第一个点是精确的原点（可能存在浮点误差） / ensure the first point is exactly the origin (may carry floating-point error)
        control_points[0] = [0, 0, 0]
        
        result = {
            'success': True,
            'method': 'cubic_spline_uniform',
            'num_points': N,
        }
        
        return control_points.astype(np.float32), result
    
    def _sample_waypoints(self, trajectory_points: np.ndarray, 
                         num_waypoints: int = 8, 
                         arc_length: float = 0.2) -> tuple:
        """
        在轨迹上按固定弧长间隔采样 waypoints。 / Sample waypoints at a fixed arc-length interval along the trajectory.

        策略：
        - 固定从起点开始，按固定 0.2m 间隔采样。
        - 若轨迹不够长，用最后一个采样点重复填充。

        Strategy:
        - Always start from the origin and sample at a fixed 0.2m interval.
        - If the trajectory is not long enough, pad with the last sampled point.

        Args:
            trajectory_points: (N, 3) GT 轨迹点（相对坐标，起点为原点） / (N, 3) GT trajectory points (relative coordinates, start at origin)
            num_waypoints: 采样点数，默认 8 / number of waypoints to sample, default 8
            arc_length: 弧长间隔，默认 0.2m / arc-length interval, default 0.2m

        Returns:
            tuple: (waypoints, fitting_result)
                - waypoints: (8, 3) 采样得到的 waypoints / (8, 3) sampled waypoints
                - fitting_result: 包含采样信息的字典 / dict with sampling information
        """
        N = len(trajectory_points)

        # 计算累积弧长 / compute cumulative arc length
        diffs = np.diff(trajectory_points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        cumulative_lengths = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative_lengths[-1]

        waypoints = np.zeros((num_waypoints, 3))

        if total_length < arc_length:
            # 极短轨迹：只有起点和终点 / extremely short trajectory: only start and end
            waypoints[0] = trajectory_points[0]
            waypoints[1:] = trajectory_points[-1]  # 剩余全部用终点填充 / pad the rest with the end point
            actual_num = 1
        else:
            # 固定从起点开始采样（总是输出前 1.4m） / always start sampling from the origin (always output the first 1.4m)
            start_length = 0.0

            # 计算实际能采样的点数 / compute how many points can actually be sampled
            available_length = total_length - start_length
            actual_num = min(num_waypoints, int(available_length / arc_length) + 1)

            # 生成目标弧长 / generate target arc lengths
            target_lengths = start_length + np.arange(actual_num) * arc_length

            # 插值采样 / interpolated sampling
            for i, target_len in enumerate(target_lengths):
                idx = np.searchsorted(cumulative_lengths, target_len)
                if idx >= N:
                    waypoints[i] = trajectory_points[-1]
                elif idx == 0:
                    waypoints[i] = trajectory_points[0]
                else:
                    # 线性插值 / linear interpolation
                    ratio = (target_len - cumulative_lengths[idx-1]) / \
                            (cumulative_lengths[idx] - cumulative_lengths[idx-1])
                    waypoints[i] = trajectory_points[idx-1] + \
                                  ratio * (trajectory_points[idx] - trajectory_points[idx-1])

            # 剩余位置用最后一个采样点填充 / pad the remaining slots with the last sampled point
            if actual_num < num_waypoints:
                waypoints[actual_num:] = waypoints[actual_num - 1]

        # 注意：trajectory_points 已是相对坐标（起点为原点），无需再次转换，可直接返回
        # Note: trajectory_points is already in relative coordinates (start at origin), so no further conversion is needed

        result = {
            'success': True,
            'method': 'waypoint_sampling',
            'arc_length': arc_length,
            'total_length': total_length,
            'actual_num': actual_num
        }
        
        return waypoints.astype(np.float32), result
    
    def _linear_fallback_control_points(self, trajectory_points: np.ndarray) -> np.ndarray:
        """
        以线性插值作为回退方案生成 8 个控制点。 / Generate 8 control points via linear interpolation as a fallback.

        Args:
            trajectory_points: (N, 3) 的三维轨迹点数组 / (N, 3) array of 3D trajectory points

        Returns:
            control_points: (8, 3) 线性插值得到的控制点数组 / (8, 3) array of linearly interpolated control points
        """
        n_points = len(trajectory_points)

        # 为原始点构造参数值 / create parameter values for original points
        t_original = np.linspace(0, 1, n_points)

        # 为 num_control_points 个控制点构造参数值 / create parameter values for num_control_points control points
        t_control = np.linspace(0, 1, self.num_control_points)

        # 逐轴插值 / interpolate each axis
        control_points = np.zeros((self.num_control_points, 3))
        for axis in range(3):
            control_points[:, axis] = np.interp(t_control, t_original, trajectory_points[:, axis])

        return control_points.astype(np.float32)

    def _apply_augmentation(self, depth_sequence: torch.Tensor) -> torch.Tensor:
        """对深度序列应用随机数据增强。 / Apply random augmentations to the depth sequence."""
        if not self.data_augmentation:
            return depth_sequence

        # 所有数据增强已被移除（水平翻转与亮度调整）。 / all augmentations have been removed (horizontal flip and brightness adjustment).
        # 可在此添加其他深度图增强方法，如： / other depth-image augmentations can be added here, such as:
        # - 高斯噪声 / Gaussian noise
        # - 深度值的小幅随机偏移 / small random offsets on depth values
        # - 局部深度置零（模拟深度相机的空洞） / zeroing local depth (simulating depth-camera holes)

        return depth_sequence
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        max_attempts = 50  # 防止死循环 / prevent infinite loops

        for attempt in range(max_attempts):
            try:
                # 根据权重随机选择场景 / randomly choose a scene according to the weights
                scene_name = self._sample_scene()
                if not scene_name or scene_name not in self.scene_runs:
                    continue

                # 从选中的场景中随机选择 run / randomly choose a run from the selected scene
                available_runs = self.scene_runs[scene_name]
                if not available_runs:
                    continue

                run_key = random.choice(available_runs)
                run_data = self.run_info[run_key]
                run_length = run_data['length']

                # 检查 run 是否足够长 / check if the run is long enough
                min_required_length = self.min_gap + 1
                if run_length < min_required_length:
                    continue

                # 随机选择起始帧 / randomly select the start frame
                max_start_idx = run_length - min(self.min_gap + 1, self.max_gap + 1)
                if max_start_idx < 0:  # run 即使按 max_gap 也太短 / run too short even for max_gap
                    continue
                start_idx = random.randint(0, max_start_idx)

                # 随机选择终止帧（距起点至少 min_gap、至多 max_gap） / randomly select the end frame (at least min_gap, at most max_gap after start)
                min_end_idx = start_idx + self.min_gap
                max_end_idx = min(start_idx + self.max_gap, run_length - 1)

                # 确保区间有效 / ensure the range is valid
                if min_end_idx > max_end_idx:
                    continue

                end_idx = random.randint(min_end_idx, max_end_idx)

                # 加载深度序列 / load the depth sequence
                depth_sequence = self._load_depth_sequence(run_key, start_idx, run_length)

                # 应用数据增强 / apply augmentations
                depth_sequence = self._apply_augmentation(depth_sequence)

                # 计算整个轨迹片段的相对位姿 / compute relative poses for the entire trajectory segment
                traj_xyz = run_data['traj_xyz']
                traj_yaw = run_data['traj_yaw']
                traj_pitch = run_data['traj_pitch']

                # 获取从起点到终点所有点的相对位姿 / get relative poses for all points from start to end
                trajectory_relative_poses = self._compute_trajectory_relative_poses(
                    traj_xyz, traj_yaw, traj_pitch, start_idx, end_idx
                )

                # 弧长截断：超过 max_arc_length 时只截断监督目标（轨迹/控制点），
                # 保留原始 end_idx，使 end_relative_pose 仍指向真实的远端 goal。
                # 训练分布：远端 goal 作为条件 + 只规划前 max_arc_length 米。
                # Arc-length truncation: when exceeding max_arc_length, only truncate the
                # supervision target (trajectory/control points) while keeping the original
                # end_idx, so end_relative_pose still points to the true distant goal.
                # Training distribution: distant goal as condition + plan only the first max_arc_length meters.
                if self.max_arc_length is not None and len(trajectory_relative_poses) >= 2:
                    diffs = np.diff(trajectory_relative_poses, axis=0)
                    seg_lengths = np.linalg.norm(diffs, axis=1)
                    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
                    if cum_lengths[-1] > self.max_arc_length:
                        cut_idx = int(np.searchsorted(cum_lengths, self.max_arc_length))
                        cut_idx = max(cut_idx, 2)  # 至少保留 2 个点 / keep at least 2 points
                        trajectory_relative_poses = trajectory_relative_poses[:cut_idx]

                # 拟合 B-spline 并得到控制点 / fit B-spline and get control points
                control_points, fitting_result = self._fit_bspline_control_points(trajectory_relative_poses)

                # 计算 B-spline 弧长 / compute B-spline arc length
                try:
                    bspline_arc_length = compute_bspline_arc_length_from_result(fitting_result)
                except Exception as e:
                    print(f"弧长计算失败: {e}")
                    bspline_arc_length = 0.0

                # 同时计算终点相对位姿以保持兼容性 / also compute the end-point relative pose for compatibility
                end_relative_pose = self._compute_relative_pose(
                    traj_xyz[start_idx], traj_yaw[start_idx], traj_pitch[start_idx],
                    traj_xyz[end_idx], traj_yaw[end_idx], traj_pitch[end_idx]
                )
                # 保留两位小数（与 B-spline 拟合的输入预处理保持一致） / round to two decimals (consistent with the input preprocessing of B-spline fitting)
                end_relative_pose = np.round(end_relative_pose.astype(np.float32), 2)

                # 获取终点的绝对坐标 / get end point absolute coordinates
                end_absolute_xyz = traj_xyz[end_idx].astype(np.float32)  # (3,) 世界绝对坐标 / absolute world coordinates
                start_absolute_xyz = traj_xyz[start_idx].astype(np.float32)  # (3,) 世界绝对坐标 / absolute world coordinates

                sample = {
                    'depth_sequence': depth_sequence,  # (sequence_length, 1, H, W)
                    'control_points': torch.from_numpy(control_points),  # (8, 3) - 固定尺寸 / fixed size!
                    'end_relative_pose': torch.from_numpy(end_relative_pose),  # (3,) 兼容用 / for compatibility
                    'relative_pose': torch.from_numpy(end_relative_pose),  # (3,) 兼容用 / for compatibility
                    'end_absolute_xyz': torch.from_numpy(end_absolute_xyz),  # (3,) 世界绝对坐标 / absolute world coordinates
                    'start_absolute_xyz': torch.from_numpy(start_absolute_xyz),  # (3,) 世界绝对坐标 / absolute world coordinates
                    'run_id': run_key,  # 现包含场景信息，如 "dataset_indoor_run_0001" / now contains scene info, e.g. "dataset_indoor_run_0001"
                    'scene_name': run_data['dataset_name'],  # 场景名称 / scene name
                    'original_run_id': run_data['run_dir'],  # 原始 run ID / original run ID
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'sequence_length': self.sequence_length,
                    'trajectory_length': len(trajectory_relative_poses),
                    'bspline_success': fitting_result.get('success', False),
                    'bspline_error': fitting_result.get('mean_error', 0.0),
                    'bspline_arc_length': bspline_arc_length  # B-spline 拟合后的总弧长 / total arc length after B-spline fitting
                }

                # 附带用于评估的 GT 稠密曲线点（优先使用拟合曲线，回退为对原轨迹线性重采样到 100 点）
                # attach dense GT curve points for evaluation (prefer the fitted curve; fall back to linearly resampling the original trajectory to 100 points)
                try:
                    fitted_points = fitting_result.get('fitted_points', None)
                    if isinstance(fitted_points, np.ndarray) and fitted_points.shape[-1] == 3:
                        gt_dense = fitted_points.astype(np.float32)
                    else:
                        # 将原始相对轨迹线性重采样到 100 个点 / linearly resample the original relative trajectory to 100 points
                        t_src = np.linspace(0, 1, len(trajectory_relative_poses))
                        t_dst = np.linspace(0, 1, 100)
                        gt_dense = np.zeros((100, 3), dtype=np.float32)
                        for ax in range(3):
                            gt_dense[:, ax] = np.interp(t_dst, t_src, trajectory_relative_poses[:, ax])
                    sample['gt_traj_fitted'] = torch.from_numpy(gt_dense)  # (100, 3)
                except Exception:
                    pass

                # 可选：对目标进行归一化（control_points 与 end_relative_pose） / optional: normalize the targets (control_points and end_relative_pose)
                if self.normalize_targets:
                    try:
                        if self._normalizer is None:
                            from sand_planner.utils.normalize import TrajectoryNormalizer
                            self._normalizer = TrajectoryNormalizer(
                                self.stats_json_path,
                                method=self.norm_method,
                                margin=self.norm_margin,
                                clamp=self.norm_clamp,
                            )
                        cp_norm = self._normalizer.normalize(control_points)  # (8, 3)
                        end_norm = self._normalizer.normalize(end_relative_pose)  # (3,)
                        sample['control_points_norm'] = torch.from_numpy(cp_norm.astype(np.float32))
                        sample['end_relative_pose_norm'] = torch.from_numpy(end_norm.astype(np.float32))
                    except Exception as ne:
                        print(f"[Warn] Target normalization failed: {ne}")

                return sample

            except Exception as e:
                if attempt == max_attempts - 1:
                    print(f"Failed to load sample after {max_attempts} attempts. Last error: {e}")
                    # 返回 dummy 样本以避免训练崩溃 / return a dummy sample to prevent training crash
                    return self._get_dummy_sample()
                continue
        
        return self._get_dummy_sample()
    
    def _sample_scene(self) -> str:
        """根据权重采样场景。 / Sample a scene according to the weights."""
        if not self.scene_weights:
            return None

        scenes = list(self.scene_weights.keys())
        weights = list(self.scene_weights.values())

        return np.random.choice(scenes, p=weights)

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """在持续加载失败时返回一个 dummy 样本。 / Return a dummy sample in case of persistent failures."""
        dummy_depth = torch.zeros(self.sequence_length, 1, *self.image_size)
        dummy_pose = torch.zeros(3)
        dummy_control_points = torch.zeros(self.num_control_points, 3)  # num_control_points 个控制点 / num_control_points control points

        return {
            'depth_sequence': dummy_depth,
            'control_points': dummy_control_points,  # (8, 3) - 固定尺寸 / fixed size!
            'end_relative_pose': dummy_pose,
            'relative_pose': dummy_pose,
            'end_absolute_xyz': dummy_pose,  # (3,) dummy 绝对坐标 / dummy absolute coordinates
            'start_absolute_xyz': dummy_pose,  # (3,) dummy 绝对坐标 / dummy absolute coordinates
            'run_id': 'dummy',
            'start_idx': 0,
            'end_idx': self.min_gap,
            'sequence_length': self.sequence_length,
            'trajectory_length': self.min_gap + 1,
            'bspline_success': False,
            'bspline_error': 0.0,
            'bspline_arc_length': 0.0  # dummy 弧长 / dummy arc length
        }


def create_bspline_dataloader(dataset_root: str,
                            batch_size: int = 16,
                            sequence_length: int = 5,
                            frame_skip: int = 2,
                            min_gap: int = 8,  # B-spline 拟合的最小值 / minimum for B-spline fitting
                            max_gap: int = 25,
                            num_workers: int = 4,
                            shuffle: bool = True,
                            pin_memory: bool = True,
                            drop_last: bool = True,
                            one_batch_only: bool = False,
                            samples_per_epoch: int = 50000,
                            **kwargs) -> DataLoader:
    """
    为输出 B-spline 控制点的 SanD-planner 数据集创建 DataLoader。 / Create a DataLoader for the SanD-planner dataset with B-spline control points.

    Args:
        dataset_root: 数据集文件夹路径 / path to dataset folder
        batch_size: 训练用的批大小 / batch size for training
        sequence_length: 历史深度帧数量 / number of historical depth frames
        frame_skip: 历史帧之间的跳帧间隔（0=连续，3=每 4 帧回取一帧） / skip interval between historical frames (0=consecutive, 3=every 4th frame back in time)
        min_gap: 起止帧之间的最小间隔（用于 B-spline 拟合） / minimum gap between start and end (for B-spline fitting)
        max_gap: 起止帧之间的最大间隔 / maximum gap between start and end
        num_workers: 工作进程数 / number of worker processes
        shuffle: 是否打乱数据 / whether to shuffle data
        pin_memory: 是否锁页内存以加速 GPU 传输 / pin memory for faster GPU transfer
        drop_last: 是否丢弃不完整的批 / drop incomplete batches
        one_batch_only: 若为 True，返回只产出恰好一个批的 DataLoader / if True, return a DataLoader that yields exactly one batch
        samples_per_epoch: 控制 __len__ 返回的样本数（影响每个 epoch 的步数 = samples_per_epoch / batch_size） / controls the number of samples returned by __len__ (steps per epoch = samples_per_epoch / batch_size)
        **kwargs: 传给 SandPlannerBSplineDataset 的额外参数 / additional arguments for SandPlannerBSplineDataset
    """
    
    base_dataset = SandPlannerBSplineDataset(
        dataset_root=dataset_root,
        sequence_length=sequence_length,
        frame_skip=frame_skip,
        min_gap=min_gap,
        max_gap=max_gap,
        samples_per_epoch=samples_per_epoch,
        **kwargs
    )
    
    if one_batch_only:
        class OneBatchWrapper(Dataset):
            """包装数据集，使其恰好产出 `batch_size` 个样本后停止。 / Wrap a dataset to produce exactly `batch_size` items, then stop.

            适用于从 DataLoader 中返回单个批。 / Useful for returning a single batch from a DataLoader.
            """
            def __init__(self, base: Dataset, count: int):
                self.base = base
                self.count = count
            def __len__(self):
                return self.count
            def __getitem__(self, idx):
                # 底层数据集是随机采样的，idx 在语义上未被使用 / underlying dataset sampling is random; idx is not used semantically
                return self.base[idx]

        dataset = OneBatchWrapper(base_dataset, batch_size)
        # 单批场景下，避免 worker/persistent 开销 / for the one-shot batch, avoid worker/persistent overhead
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=collate_bspline_batch,
        )

    dataset = base_dataset
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        collate_fn=collate_bspline_batch  # 使用自定义 collate 函数 / use custom collate function
    )


# 简单的 collate 函数 —— 无需 padding！ / simple collate function - no padding needed!
def collate_bspline_batch(batch):
    """用于 B-spline 控制点的自定义 collate 函数 —— 无需 padding！ / Custom collate function for B-spline control points - no padding needed!"""
    depth_sequences = torch.stack([item['depth_sequence'] for item in batch])
    control_points = torch.stack([item['control_points'] for item in batch])  # (B, 8, 3)
    end_relative_poses = torch.stack([item['end_relative_pose'] for item in batch])
    end_absolute_xyz = torch.stack([item['end_absolute_xyz'] for item in batch])  # (B, 3)
    start_absolute_xyz = torch.stack([item['start_absolute_xyz'] for item in batch])  # (B, 3)

    out = {
        'depth_sequence': depth_sequences,        # (B, seq_len, 1, H, W)
        'control_points': control_points,         # (B, 8, 3) - 固定尺寸，无需 padding / fixed size, no padding!
        'end_relative_pose': end_relative_poses,  # (B, 3)
        'relative_pose': end_relative_poses,      # (B, 3) - 兼容用 / for compatibility
        'end_absolute_xyz': end_absolute_xyz,     # (B, 3) - 终点的世界绝对坐标 / absolute world coordinates of end point
        'start_absolute_xyz': start_absolute_xyz, # (B, 3) - 起点的世界绝对坐标 / absolute world coordinates of start point
        'run_ids': [item['run_id'] for item in batch],  # 包含场景信息的完整 run ID / full run IDs with scene info
        'scene_names': [item['scene_name'] for item in batch],  # 场景名称 / scene names
        'original_run_ids': [item['original_run_id'] for item in batch],  # 原始 run ID / original run IDs
        'start_indices': torch.tensor([item['start_idx'] for item in batch]),
        'end_indices': torch.tensor([item['end_idx'] for item in batch]),
        'trajectory_lengths': torch.tensor([item['trajectory_length'] for item in batch]),
        'bspline_success': torch.tensor([item['bspline_success'] for item in batch]),
        'bspline_errors': torch.tensor([item['bspline_error'] for item in batch]),
        'bspline_arc_lengths': torch.tensor([item['bspline_arc_length'] for item in batch])  # B-spline 弧长 / B-spline arc lengths
    }

    # 可选：打包归一化字段 / optional: pack the normalized fields
    if 'control_points_norm' in batch[0]:
        out['control_points_norm'] = torch.stack([item['control_points_norm'] for item in batch])
    if 'end_relative_pose_norm' in batch[0]:
        out['end_relative_pose_norm'] = torch.stack([item['end_relative_pose_norm'] for item in batch])
    if 'gt_traj_fitted' in batch[0]:
        out['gt_traj_fitted'] = torch.stack([item['gt_traj_fitted'] for item in batch])  # (B, 100, 3)

    return out


if __name__ == "__main__":
    # 快速测试 / quick test
    print("Testing Multi-Scene SanD-planner B-Spline DataLoader...")

    dataset_root = "dataset"

    # 测试 1：加载所有场景 / test 1: load all scenes
    print("\n=== Test 1: All Scenes ===")
    dataloader_all = create_bspline_dataloader(
        dataset_root,
        batch_size=4,
        sequence_length=5,
        min_gap=8,  # B-spline 的最小值 / minimum for B-spline
        max_gap=25,
        image_size=(480, 640),
        num_workers=0,  # 调试时用 0 / use 0 for debugging
        samples_per_epoch=100
    )

    # 测试 2：只加载特定场景 / test 2: load only specific scenes
    print("\n=== Test 2: Specific Scenes (indoor, stair) ===")
    dataloader_specific = create_bspline_dataloader(
        dataset_root,
        batch_size=4,
        sequence_length=5,
        min_gap=8,
        max_gap=25,
        image_size=(480, 640),
        num_workers=0,
        samples_per_epoch=50,
        dataset_types=['indoor', 'stair']  # 只加载包含这些关键词的数据集 / load only datasets matching these keywords
    )

    # 测试 3：自定义场景权重 / test 3: custom scene weights
    print("\n=== Test 3: Custom Scene Weights ===")
    dataloader_weighted = create_bspline_dataloader(
        dataset_root, 
        batch_size=4, 
        sequence_length=5,
        min_gap=8,
        max_gap=25,
        image_size=(480, 640),
        num_workers=0,
        samples_per_epoch=50,
        scene_sampling_weights={
            'dataset_indoor': 0.4,
            'dataset_stair': 0.3,
            'dataset_campus': 0.3
        }
    )
    
    print("\nLoading first batch from all scenes...")
    try:
        batch = next(iter(dataloader_all))
        print(f"✓ Depth sequence shape: {batch['depth_sequence'].shape}")
        print(f"✓ Control points shape: {batch['control_points'].shape}")
        print(f"✓ End relative pose shape: {batch['end_relative_pose'].shape}")
        print(f"✓ Scene names: {batch['scene_names']}")
        print(f"✓ Run IDs: {batch['run_ids']}")
        print(f"✓ Original run IDs: {batch['original_run_ids']}")
        print(f"✓ B-spline success rate: {batch['bspline_success'].float().mean():.2f}")

        # 统计场景分布 / tally the scene distribution
        from collections import Counter
        scene_counts = Counter(batch['scene_names'])
        print(f"✓ Scene distribution in batch: {dict(scene_counts)}")
        
        print("\nMulti-Scene DataLoader test successful!")
        print("\n🎉 New features:")
        print("  • Multi-scene support: automatically loads all dataset_* folders")
        print("  • Scene filtering: can specify which scenes to use")
        print("  • Scene weighting: can control sampling probability per scene")
        print("  • Rich metadata: scene names, original run IDs, etc.")
        print("  • Pitch support: full 6DOF pose calculation")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
