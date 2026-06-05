#!/usr/bin/env python3
"""
SanD-planner 简易推理服务器 / Simple SanD-planner inference server.

无频率限制，每次请求都重新规划，只支持 point goal（点目标）导航。
No rate limit; every request triggers a fresh planning pass, and only point-goal navigation is supported.
"""

from PIL import Image
from flask import Flask, request, jsonify
from sand_planner.config import InferenceConfig
from sand_planner.agent.sand_planner_agent import SandPlannerAgent
import numpy as np
import cv2
import imageio
import time
import datetime
import json
import os

# 设置 matplotlib 使用非交互式后端，避免 GUI 相关的错误
# Configure matplotlib to use a non-interactive backend to avoid GUI-related errors.
import matplotlib
# 使用 Anti-Grain Geometry 后端，无需 X11 或其他 GUI
# Use the Anti-Grain Geometry backend, which requires neither X11 nor any other GUI.
matplotlib.use('Agg')

from PIL import Image, ImageDraw, ImageFont
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8890, help="服务器端口")
parser.add_argument("--checkpoint", type=str, default=None, help="模型checkpoint路径，默认使用InferenceConfig中的配置")
args = parser.parse_known_args()[0]
# 备选 checkpoint / alt checkpoint: best_model_cubic_spline.pth
app = Flask(__name__)
navigator = None
fps_writer = None
depth_save_dirs = {}  # 每个环境的保存目录 / per-environment save directory {env_id: dir_path}
depth_step_counters = {}  # 每个环境的步数计数器 / per-environment step counter {env_id: step_count}

@app.route("/navigator_reset", methods=['POST'])
def navigator_reset():
    """重置导航器 / Reset the navigator."""
    global navigator, fps_writer

    intrinsic = np.array(request.get_json().get('intrinsic'))
    threshold = np.array(request.get_json().get('stop_threshold'))
    batchsize = np.array(request.get_json().get('batch_size'))

    print('相机内参:', intrinsic)
    print('停止阈值:', threshold)
    print('批处理大小:', batchsize)

    if navigator is None:
        print("🚀 创建SanD-planner Agent...")
        config_kwargs = dict(
            device='cuda:0',
            save_visualizations=False,
            save_data=False,
            show_verbose=False,
        )
        if args.checkpoint is not None:
            config_kwargs['checkpoint_path'] = args.checkpoint
        config = InferenceConfig(**config_kwargs)
        navigator = SandPlannerAgent(
            image_intrinsic=intrinsic,
            config=config,
            verbose=True,
        )
        navigator.reset(batchsize, threshold)
        # 预热：触发 torch.compile 编译，避免首次推理 timeout
        # Warm-up: trigger torch.compile so the first real inference does not time out.
        print("🔥 预热 torch.compile (首次编译约15秒)...")
        warmup_start = time.time()
        dummy_img = np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8)
        dummy_dep = np.random.rand(1, 480, 640, 1).astype(np.float32) * 5.0
        dummy_goal = np.array([[3.0, 0.0, 0.0]])
        for i in range(3):
            navigator.step_pointgoal(dummy_goal, dummy_img, dummy_dep)
        navigator.reset(batchsize, threshold)
        print(f"✅ 预热完成 ({time.time()-warmup_start:.1f}秒)")
    else:
        print("🔄 重置现有SanD-planner Agent...")
        # 更新相机内参 / Update the camera intrinsics.
        if not np.array_equal(navigator.image_intrinsic, intrinsic):
            print("📷 更新相机内参...")
            navigator.image_intrinsic = intrinsic
            navigator.update_camera_config(intrinsic)
        navigator.reset(batchsize, threshold)

    # 初始化 FPS 写入器 / Initialize the FPS video writer.
    if fps_writer is None:
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        fps_writer = imageio.get_writer("{}_fps_sand_planner_simple.mp4".format(format_time), fps=7)
    else:
        fps_writer.close()
        format_time = datetime.datetime.fromtimestamp(time.time())
        format_time = format_time.strftime("%Y-%m-%d %H:%M:%S")
        fps_writer = imageio.get_writer("{}_fps_sand_planner_simple.mp4".format(format_time), fps=7)

    return jsonify({"algo": "sand_planner_simple"})

@app.route("/navigator_reset_env", methods=['POST'])
def navigator_reset_env():
    """重置特定环境 / Reset a specific environment."""
    global navigator, depth_save_dirs, depth_step_counters
    env_id = int(request.get_json().get('env_id'))
    navigator.reset_env(env_id)

    # 创建新的保存文件夹 / Create a fresh save folder.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"depth_images/env_{env_id}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    depth_save_dirs[env_id] = save_dir
    depth_step_counters[env_id] = 0

    print(f"🔄 重置环境 {env_id}")
    print(f"📁 创建新的depth保存目录: {save_dir}")
    return jsonify({"algo": "sand_planner_simple", "depth_save_dir": save_dir})

@app.route("/pointgoal_step", methods=['POST'])
def pointgoal_step():
    """点目标导航步进（无频率限制，每次请求都重新规划）/ Point-goal navigation step (no rate limit; re-plans on every request)."""
    global navigator, fps_writer, depth_save_dirs, depth_step_counters

    start_time = time.time()

    # 解析输入数据（图像、深度图、目标点）/ Parse the input data (image, depth, goal).
    image_file = request.files['image']
    depth_file = request.files['depth']
    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)
    batch_size = navigator.batch_size

    # 获取 env_id（如果存在）/ Get env_id (if present).
    env_id = goal_data.get('env_id', 0) if isinstance(goal_data, dict) else 0

    phase1_time = time.time()

    # 处理图像数据 / Process the image data.
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image_original = np.asarray(image)  # 保留原始图像以便存盘 / Keep the original image for saving.
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    # 处理深度图数据 / Process the depth image data.
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    # 保存深度图与 RGB 图像；若该 env_id 尚无保存目录则自动创建。
    # Save the depth image and the RGB image; auto-create a save directory if this env_id has none yet.
    if env_id not in depth_save_dirs:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"depth_images/env_{env_id}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        depth_save_dirs[env_id] = save_dir
        depth_step_counters[env_id] = 0
        print(f"📁 自动创建depth保存目录: {save_dir}")

    save_dir = depth_save_dirs[env_id]
    step_count = depth_step_counters[env_id]

    # 保存每个环境的深度图与 RGB / Save the depth image and RGB image for each environment.
    for i in range(batch_size):
        # 保存深度图——原始 uint16 格式（用于精确数据）
        # Save the depth image in the raw uint16 format (for precise data).
        depth_raw = (depth[i, :, :, 0] * 10000.0).astype(np.uint16)
        depth_raw_path = os.path.join(save_dir, f"depth_{step_count:04d}.png")
        cv2.imwrite(depth_raw_path, depth_raw)

        # 保存深度图——可视化格式（灰度图，可直接查看）
        # Save the depth image in a visualization format (grayscale, directly viewable).
        depth_vis = depth[i, :, :, 0]
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8) * 255.0
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis_path = os.path.join(save_dir, f"depth_vis_{step_count:04d}.png")
        cv2.imwrite(depth_vis_path, depth_vis)

        # 保存 RGB——直接保存原始图像，最高质量
        # Save the RGB image directly from the original image at the highest quality.
        if batch_size == 1:
            # 单环境：直接保存原始图像 / Single environment: save the original image directly.
            rgb_path = os.path.join(save_dir, f"rgb_{step_count:04d}.png")
            cv2.imwrite(rgb_path, cv2.cvtColor(image_original, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            # 多环境：保存对应的图像块 / Multiple environments: save the corresponding image tile.
            rgb_img = image[i]  # 已经是 BGR 格式 / Already in BGR format.
            rgb_path = os.path.join(save_dir, f"rgb_{step_count:04d}.png")
            cv2.imwrite(rgb_path, rgb_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    depth_step_counters[env_id] += 1

    phase2_time = time.time()

    # 执行导航推理——每次都重新规划，无频率限制。
    # Run navigation inference: re-plan on every call, with no rate limit.
    try:
        execute_trajectory, all_trajectory, all_values, trajectory_mask = navigator.step_pointgoal(goal, image, depth)
        phase3_time = time.time()

        # 写入 FPS 视频 / Write a frame to the FPS video.
        if fps_writer is not None and trajectory_mask is not None:
            fps_writer.append_data(trajectory_mask)
        phase4_time = time.time()

        # 打印各阶段计时统计 / Print per-phase timing statistics.
        all_time = time.time() - start_time
        print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f  (%.0fHz)" % (
            phase1_time - start_time,
            phase2_time - phase1_time,
            phase3_time - phase2_time,
            phase4_time - phase3_time,
            all_time,
            1.0 / max(all_time, 0.001)
        ))
        # 写入 timing 日志供后续分析 / Write the timing log for later analysis.
        with open("timing.log", "a") as f:
            f.write("%.4f %.4f %.4f %.4f %.4f\n" % (
                phase1_time - start_time,
                phase2_time - phase1_time,
                phase3_time - phase2_time,
                phase4_time - phase3_time,
                all_time
            ))

        # 验证返回数据 / Validate the returned data.
        if execute_trajectory is None or all_trajectory is None or all_values is None:
            raise ValueError("SanD-planner返回了None值")

        # 返回结果：execute_trajectory / all_trajectory / all_values 三个字段
        # Return the result with the execute_trajectory / all_trajectory / all_values fields.
        return jsonify({
            'trajectory': execute_trajectory.tolist() if hasattr(execute_trajectory, 'tolist') else execute_trajectory,
            'all_trajectory': all_trajectory.tolist() if hasattr(all_trajectory, 'tolist') else all_trajectory,
            'all_values': all_values.tolist() if hasattr(all_values, 'tolist') else all_values
        })

    except Exception as e:
        phase3_time = time.time()
        phase4_time = time.time()

        print(f"❌ SanD-planner推理失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()

        # 打印计时统计 / Print timing statistics.
        print("phase1:%f, phase2:%f, phase3:%f, phase4:%f, all:%f" % (
            phase1_time - start_time,
            phase2_time - phase1_time,
            phase3_time - phase2_time,
            phase4_time - phase3_time,
            time.time() - start_time
        ))

        # 返回错误状态，不回退到默认轨迹 / Return an error status; do not fall back to a default trajectory.
        return jsonify({
            'status': 'error',
            'message': f'SanD-planner推理失败: {str(e)}',
            'error_type': type(e).__name__
        }), 500

@app.route("/health", methods=['GET'])
def health_check():
    """健康检查接口 / Health-check endpoint."""
    global navigator
    return jsonify({
        "status": "healthy",
        "agent_loaded": navigator is not None,
        "algorithm": "sand_planner_simple",
        "checkpoint": args.checkpoint
    })

if __name__ == "__main__":
    preview_config = InferenceConfig() if args.checkpoint is None else InferenceConfig(checkpoint_path=args.checkpoint)
    print("🚀 启动Simple SanD-planner服务器...")
    print(f"📍 端口: {args.port}")
    print(f"🤖 模型: {preview_config.checkpoint_path}")
    print(f"✨ 特性: 无频率限制，每次都重新规划")
    print(f"🔗 仅支持点目标（point-goal）导航")
    print(f"🌐 服务器运行在 http://127.0.0.1:{args.port}")

    app.run(host='127.0.0.1', port=args.port)
