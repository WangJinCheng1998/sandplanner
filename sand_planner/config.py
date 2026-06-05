"""SanD-planner 推理流水线的统一配置 / Unified configuration for the SanD-planner inference pipeline.

集中管理 agent、evaluator、orchestrator 等模块共用的参数。
Centralizes the parameters shared across the agent, evaluator, and orchestrator modules.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from sand_planner.utils.nvblox_esdf import NvbloxESDFMapper
    NVBLOX_AVAILABLE = True
except ImportError:
    NVBLOX_AVAILABLE = False


@dataclass
class InferenceConfig:
    """推理配置类 / Inference configuration."""

    # 解析相对路径(stats_path、数据目录等)的基准目录;
    # 可用环境变量 SAND_PLANNER_BASE_DIR 覆盖,默认取仓库根目录。
    # Base directory for resolving relative paths (stats_path, data dirs, ...);
    # override via the SAND_PLANNER_BASE_DIR env var; defaults to the repo root.
    base_dir: str = os.environ.get(
        "SAND_PLANNER_BASE_DIR",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    # 路径配置（使用相对路径） / Path configuration (relative paths)
    checkpoint_path: str = "checkpoints/Max_2.1m.pth"
    stats_path: str = "data/outputs/trajectory_stats_mg2_mg50_2d.json"
    input_depth_dir: str = "data/real_test/draw"
    consecutive_depth_dir: str = "data/real_test/real_lab_converted/depth"
    output_dir: str = "outputs/inference_results"

    # 模型配置 / Model configuration
    sequence_length: int = 4
    fusion_strategy: str = 'concat'  # 可选值/options: 'concat', 'average', 'attention'
    use_consecutive_frames: bool = True
    trajectory_interpolation: str = 'bspline'  # 可选值/options: 'bspline' 或/or 'cubic_spline'
    prediction_mode: str = 'control_points'  # 可选值/options: 'control_points' 或/or 'waypoints'
    num_control_points: int = 8  # B-spline 控制点数量（必须与训练时一致） / Number of B-spline control points (must match training)
    # 严格模式：与 checkpoint 不符或 checkpoint 未记录时直接报错
    # Strict mode: raise an error if the value mismatches the checkpoint or is not recorded in it
    strict_num_control_points: bool = False
    num_transformer_layers: int = 2
    num_heads: int = 4

    # 推理配置 / Inference configuration
    target_position: List[float] = field(default_factory=lambda: [4.0, 1.0, 0.0])
    batch_size: int = 32
    num_inference_steps: int = 10
    device: str = 'cuda'
    enable_warm_start: bool = True
    warm_start_resume_step: int = 5  # 选择重新加噪对应的 scheduler 步编号 / Scheduler timestep index at which to re-inject noise

    # Best Plan Candidate Backtracking 配置 / Best plan candidate backtracking configuration
    enable_plan_backtracking: bool = False  # 启用最优计划回溯 / Enable best-plan backtracking
    backtracking_bonus: float = 0.01  # 给旧轨迹的奖励系数 / Bonus weight given to the previous trajectory
    execution_distance_per_frame: float = 0.1  # 每帧执行的距离（米） / Distance executed per frame (meters)

    # 图像处理配置 / Image processing configuration
    image_height: int = 168
    image_width: int = 224
    downscale_depth: bool = True
    max_depth: float = 8.0

    # 相机参数 / Camera parameters
    camera_fx: float = 389.551
    camera_fy: float = 389.551
    camera_ppx: float = 324.211
    camera_ppy: float = 235.656

    # ESDF 配置 - 2D 平面查询版本（适合地面机器人）
    # ESDF configuration - 2D planar query variant (suited for ground robots)
    use_nvblox: bool = False  # 禁用 NVBlox，改用 CPU 方法（规避 GPU 内存问题） / Disable NVBlox, use CPU method (avoids GPU memory issues)
    use_gpu_esdf: bool = True  # 使用 GPU (CuPy) 加速 EDT 计算 / Use GPU (CuPy) to accelerate EDT computation
    esdf_voxel_size: float = 0.05  # 保持高精度 / Keep high resolution
    # 体素栅格尺寸：X 约 4m（左右）、Y 约 2m（上下）、Z 约 5m（前后），已针对内存优化
    # Voxel grid size: X ~4m (lateral), Y ~2m (vertical), Z ~5m (forward), tuned for memory
    # 备选/alt: (40, 30, 40)
    esdf_grid_size: Tuple[int, int, int] = (80, 60, 100)
    esdf_grid_origin: Tuple[float, float, float] = (-2.0, -1.5, 0.0)  # 居中覆盖轨迹范围 / Centered to cover the trajectory range
    esdf_downsample_factor: int = 1  # 平衡性能 / Balance performance
    esdf_surface_threshold: float = 0.2  # 障碍物膨胀半径（≈机器人半径），用作安全距离 / Obstacle inflation radius (≈robot radius), used as clearance
    clearance_height: Optional[float] = 0.36  # 查询离地约 0.5m 的 2D 平面（机器人胸部高度） / 2D query plane ~0.5m above ground (robot chest height)

    # 可视化配置 / Visualization configuration
    save_visualizations: bool = True
    save_data: bool = True
    show_verbose: bool = False

    # Agent 参数（此前在 SandPlannerAgent 中硬编码）
    # Agent parameters (previously hardcoded in SandPlannerAgent)
    predict_size: int = 24
    default_behavior: str = "stop"
    depth_cache_size: int = 4
    mapper_reset_interval: int = 50
    esdf_safety_threshold: float = -100.0

    # 轨迹评估参数（此前在 evaluator/orchestrator 中硬编码）
    # Trajectory evaluation parameters (previously hardcoded in evaluator/orchestrator)
    clearance_max_points: int = 40
    eval_weight_clear: float = 10000.0
    eval_weight_length: float = 0.0
    eval_weight_goal: float = 0.0001
    eval_safety_margin: float = 0.25
    arc_length_step: float = 0.1

    def __post_init__(self):
        """初始化默认值和路径 / Initialize default values and paths."""
        # 将相对路径转换为绝对路径 / Resolve relative paths into absolute paths
        self.checkpoint_path = os.path.join(self.base_dir, self.checkpoint_path)
        self.stats_path = os.path.join(self.base_dir, self.stats_path)
        self.input_depth_dir = os.path.join(self.base_dir, self.input_depth_dir)
        self.consecutive_depth_dir = os.path.join(self.base_dir, self.consecutive_depth_dir)
        self.output_dir = os.path.join(self.base_dir, self.output_dir)

        # 创建输出目录 / Create the output directory
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def camera_intrinsics(self) -> Dict[str, float]:
        """返回相机内参字典 / Return the camera intrinsics dictionary."""
        return {
            'fx': self.camera_fx, 'fy': self.camera_fy,
            'ppx': self.camera_ppx, 'ppy': self.camera_ppy
        }

    @property
    def esdf_config(self) -> Dict[str, Any]:
        """返回 ESDF 配置字典 / Return the ESDF configuration dictionary."""
        base_config = {
            'voxel_size': self.esdf_voxel_size,
            'grid_size': self.esdf_grid_size,
            'grid_origin': self.esdf_grid_origin,
        }

        # CPU 方法需要额外参数 / The CPU method requires extra parameters
        if not self.use_nvblox or not NVBLOX_AVAILABLE:
            base_config.update({
                'downsample_factor': self.esdf_downsample_factor,
                'surface_threshold': self.esdf_surface_threshold,
                'use_gpu': self.use_gpu_esdf  # 使用 GPU (CuPy) 加速 EDT / Use GPU (CuPy) to accelerate EDT
            })

        return base_config
