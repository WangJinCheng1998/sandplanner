#!/usr/bin/env python3
"""
肘部测试（elbow test）：对 ~6m 的真实 GT 轨迹段，用不同数量的控制点拟合
clamped cubic B-spline，统计重建误差（mean / max）随控制点数 m 的变化，
找出"再加控制点也不明显降误差"的拐点。

Elbow test: for ~6m real GT trajectory segments, fit a clamped cubic B-spline
with varying numbers of control points and track how the reconstruction error
(mean / max) changes with the control point count m, locating the knee where
adding more control points no longer meaningfully reduces the error.

与训练 pipeline 保持一致：
  - 拟合前坐标保留两位小数 (round 2)
  - 等弧长重采样后用 clamped cubic B-spline 最小二乘解控制点 (端点硬约束)
误差评估：在重采样点的弦长参数上比较拟合曲线与重采样点（mean / max 欧氏距离）。

Consistent with the training pipeline:
  - round coordinates to two decimals before fitting (round 2)
  - after equal-arc-length resampling, solve the control points via least squares
    for the clamped cubic B-spline (hard endpoint constraints)
Error evaluation: compare the fitted curve against the resampled points at the
chord-length parameters of those points (mean / max Euclidean distance).
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy.interpolate import BSpline

# 直接按文件路径加载 bspline.py，绕过 sand_planner 包的 __init__（它会 import cv2 等重依赖）
# Load bspline.py directly by file path, bypassing the sand_planner package __init__
# (which would import heavy dependencies such as cv2).
import importlib.util
_bspline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'sand_planner', 'utils', 'bspline.py')
_spec = importlib.util.spec_from_file_location('_bspline_standalone', _bspline_path)
_bspline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bspline)
resample_equal_arclength = _bspline.resample_equal_arclength
_chord_length_param = _bspline._chord_length_param
_solve_control_points = _bspline._solve_control_points


def fit_mcp(segment_xyz: np.ndarray, m: int, k: int = 3,
            resample_points: int = 64) -> tuple:
    """用 m 个控制点拟合 clamped cubic B-spline，返回 (mean_err, max_err)。 /
    Fit a clamped cubic B-spline with m control points and return (mean_err, max_err)."""
    Q = np.round(np.asarray(segment_xyz, dtype=float), 2)
    Q = resample_equal_arclength(Q, resample_points)
    u = _chord_length_param(Q)
    P, t, _ = _solve_control_points(
        Q, u, k=k, m=m, knot_strategy='quantile',
        enforce_endpoints=True, ridge_lambda=1e-6,
    )
    bs = [BSpline(t, P[:, ax], k, extrapolate=False) for ax in range(3)]
    fit = np.stack([bs[ax](u) for ax in range(3)], axis=-1)
    err = np.linalg.norm(Q - fit, axis=1)
    return float(np.mean(err)), float(np.max(err))


def collect_segments(dataset_root: str, target_len: float, tol: float,
                     stride: int, max_segments: int) -> list:
    """从所有 run 的 traj_xyz.npy 中切出弧长 ≈ target_len 的段。 /
    Cut out segments with arc length ≈ target_len from the traj_xyz.npy of every run."""
    segments = []
    datasets = [d for d in sorted(os.listdir(dataset_root))
                if os.path.isdir(os.path.join(dataset_root, d))]
    for ds in datasets:
        ds_path = os.path.join(dataset_root, ds)
        runs = [r for r in sorted(os.listdir(ds_path))
                if os.path.isdir(os.path.join(ds_path, r))]
        for r in runs:
            p = os.path.join(ds_path, r, 'traj_xyz.npy')
            if not os.path.exists(p):
                continue
            try:
                traj = np.load(p).astype(float)
            except Exception:
                continue
            if traj.ndim != 2 or traj.shape[0] < 8 or traj.shape[1] != 3:
                continue
            seg_len = np.linalg.norm(np.diff(traj, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            n = len(traj)
            for s in range(0, n - 1, stride):
                # 找到第一个使弧长超过 target 的 end
                # Find the first end index whose arc length exceeds target.
                need = cum[s] + target_len
                e = int(np.searchsorted(cum, need))
                if e >= n:
                    break
                actual = cum[e] - cum[s]
                if abs(actual - target_len) <= tol and (e - s) >= 6:
                    segments.append(traj[s:e + 1].copy())
                    if len(segments) >= max_segments:
                        return segments
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root',
                    default=os.environ.get('DATASET_ROOT', 'dataset'))
    ap.add_argument('--target_len', type=float, default=6.0)
    ap.add_argument('--tol', type=float, default=0.5)
    ap.add_argument('--stride', type=int, default=20)
    ap.add_argument('--max_segments', type=int, default=2000)
    ap.add_argument('--m_list', type=int, nargs='+',
                    default=[4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 23])
    ap.add_argument('--out', default='elbow_test_6m.png')
    args = ap.parse_args()

    print(f"📂 数据根目录: {args.dataset_root}")
    print(f"🎯 目标段长: {args.target_len}m ± {args.tol}m")
    segs = collect_segments(args.dataset_root, args.target_len, args.tol,
                            args.stride, args.max_segments)
    print(f"✅ 收集到 {len(segs)} 个 ~{args.target_len}m 轨迹段")
    if len(segs) < 10:
        print("⚠️ 段太少，结果不可靠。可调大 --tol 或调小 --stride。")
        if len(segs) == 0:
            return

    actual_lens = [float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1))) for s in segs]
    print(f"   实际段长: 均值 {np.mean(actual_lens):.2f}m, "
          f"范围 [{np.min(actual_lens):.2f}, {np.max(actual_lens):.2f}]m")

    # m -> （平均 mean 误差, 平均 max 误差）
    # m -> (mean_of_mean_err, mean_of_max_err)
    results = {}
    for m in args.m_list:
        me, mx = [], []
        for s in segs:
            try:
                a, b = fit_mcp(s, m)
                me.append(a)
                mx.append(b)
            except Exception:
                continue
        results[m] = (float(np.mean(me)), float(np.mean(mx)))

    # 打印表格 + 边际收益
    # Print the table plus the marginal gain.
    print("\n=== 重建误差 vs 控制点数 (cubic B-spline, " + f"{args.target_len}m 段) ===")
    print(f"{'m(CP)':>6} | {'段数':>9} | {'mean_err(m)':>12} | {'max_err(m)':>11} | "
          f"{'Δmean vs 前一个':>16}")
    print("-" * 70)
    prev = None
    ms = sorted(results.keys())
    for m in ms:
        mean_e, max_e = results[m]
        if prev is None:
            delta = "—"
        else:
            d = prev - mean_e
            pct = (d / prev * 100) if prev > 0 else 0.0
            delta = f"-{d*1000:.2f}mm ({pct:.1f}%)"
        seg_per_cp = args.target_len / max(m - 1, 1)
        print(f"{m:>6} | {len(segs):>9} | {mean_e*1000:>10.2f}mm | {max_e*1000:>9.2f}mm | {delta:>16}")
        prev = mean_e

    # 简单的肘部判定：边际下降首次低于"首段总下降的 5%/步"阈值
    # Simple elbow detection: the first point where the marginal drop falls below
    # the "5% of the total drop per step" threshold.
    mean_curve = np.array([results[m][0] for m in ms])
    if len(ms) >= 3:
        total_drop = mean_curve[0] - mean_curve[-1]
        # 每加一组控制点，降幅小于总降幅的 5% 即视为进入平台
        # Each time control points are added, a drop below 5% of the total drop
        # is treated as having reached the plateau.
        thresh = 0.05 * total_drop
        elbow = ms[-1]
        for i in range(1, len(ms)):
            if (mean_curve[i - 1] - mean_curve[i]) < thresh:
                elbow = ms[i - 1]
                break
        print(f"\n📍 估计肘部 (边际降幅<总降幅5% 的首个点): m ≈ {elbow} 个控制点")
        print(f"   → 对 {args.target_len}m 路径，约 {elbow} 个控制点后收益明显递减")

    # 画图
    # Plot the curve.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax1.plot(ms, mean_curve * 1000, 'o-', color='tab:blue', label='mean error')
        ax1.plot(ms, [results[m][1] * 1000 for m in ms], 's--',
                 color='tab:orange', label='max error')
        ax1.set_xlabel('number of control points (m)')
        ax1.set_ylabel('reconstruction error (mm)')
        ax1.set_title(f'Elbow test: B-spline reconstruction error vs #CP '
                      f'({args.target_len}m paths, n={len(segs)})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        print(f"\n💾 已保存曲线图: {os.path.abspath(args.out)}")
    except Exception as e:
        print(f"⚠️ 画图失败（不影响表格结果）: {e}")


if __name__ == '__main__':
    main()
