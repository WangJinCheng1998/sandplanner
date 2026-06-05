#!/usr/bin/env python3
"""
诊断 max_arc_length=6m 时"大转弯困难"的成因之一 / Diagnose one cause of "hard large turns" at max_arc_length=6m.

在固定控制点数下，检验大转弯段的 B-spline 重建误差是否显著恶化
（即大弯是否在标签里被拉直）。
按"段内净航向变化（net heading change，度）"把 ~6m 段分桶，
对每桶在不同控制点数 m 下统计重建误差的 mean/max。

Under a fixed number of control points, check whether the B-spline reconstruction
error of large-turn segments degrades significantly (i.e. whether large turns are
straightened out in the labels). Bucket the ~6m segments by their net heading change
(in degrees), and for each bucket report the mean/max reconstruction error across
different control-point counts m.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.interpolate import BSpline
import importlib.util

_bp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'sand_planner', 'utils', 'bspline.py')
_spec = importlib.util.spec_from_file_location('_bspline_standalone', _bp)
_bspline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bspline)
resample_equal_arclength = _bspline.resample_equal_arclength
_chord_length_param = _bspline._chord_length_param
_solve_control_points = _bspline._solve_control_points

DATASET_ROOT = os.environ.get('DATASET_ROOT', 'dataset')
TARGET_LEN = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
TOL, STRIDE, MAX_SEG = 0.5, 15, 4000
M_LIST = [8, 10, 12]


def fit_err(seg, m, k=3, rs=64):
    Q = resample_equal_arclength(np.round(seg.astype(float), 2), rs)
    u = _chord_length_param(Q)
    P, t, _ = _solve_control_points(Q, u, k=k, m=m, knot_strategy='quantile',
                                    enforce_endpoints=True, ridge_lambda=1e-6)
    fit = np.stack([BSpline(t, P[:, ax], k, extrapolate=False)(u) for ax in range(3)], -1)
    e = np.linalg.norm(Q - fit, axis=1)
    return float(np.mean(e)), float(np.max(e))


def net_heading_change_deg(seg):
    """段首方向与段尾方向之间的净夹角（度，在 xy 平面计算） / Net angle (degrees) between the segment's start and end directions, computed in the xy plane."""
    d = np.diff(seg[:, :2], axis=0)
    n = np.linalg.norm(d, axis=1)
    d = d[n > 1e-6]
    if len(d) < 2:
        return 0.0
    h0 = np.arctan2(d[0, 1], d[0, 0])
    h1 = np.arctan2(d[-1, 1], d[-1, 0])
    dh = np.abs(np.degrees(np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0)))
                )
    return float(dh)


def collect():
    segs = []
    for ds in sorted(os.listdir(DATASET_ROOT)):
        dsp = os.path.join(DATASET_ROOT, ds)
        if not os.path.isdir(dsp):
            continue
        for r in sorted(os.listdir(dsp)):
            p = os.path.join(dsp, r, 'traj_xyz.npy')
            if not os.path.exists(p):
                continue
            try:
                traj = np.load(p).astype(float)
            except Exception:
                continue
            if traj.ndim != 2 or traj.shape[0] < 8 or traj.shape[1] != 3:
                continue
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))])
            n = len(traj)
            for s in range(0, n - 1, STRIDE):
                e = int(np.searchsorted(cum, cum[s] + TARGET_LEN))
                if e >= n:
                    break
                if abs((cum[e] - cum[s]) - TARGET_LEN) <= TOL and (e - s) >= 6:
                    segs.append(traj[s:e + 1].copy())
                    if len(segs) >= MAX_SEG:
                        return segs
    return segs


def main():
    segs = collect()
    print(f"✅ {len(segs)} 个 ~{TARGET_LEN}m 段")
    turns = np.array([net_heading_change_deg(s) for s in segs])
    # 按净航向变化分桶 / Bucket by net heading change
    buckets = [('直行 (<20°)', turns < 20),
               ('中弯 (20-60°)', (turns >= 20) & (turns < 60)),
               ('大弯 (>=60°)', turns >= 60)]
    print(f"转弯分布: 直行 {np.sum(turns<20)}, 中弯 {np.sum((turns>=20)&(turns<60))}, "
          f"大弯 {np.sum(turns>=60)}  (最大净转角 {turns.max():.0f}°)")

    print(f"\n{'桶':>14} | {'段数':>6} | " +
          " | ".join([f"m={m} mean/max(mm)" for m in M_LIST]))
    print("-" * 80)
    for name, mask in buckets:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        cols = []
        for m in M_LIST:
            me, mx = [], []
            for i in idx:
                try:
                    a, b = fit_err(segs[i], m)
                    me.append(a); mx.append(b)
                except Exception:
                    pass
            cols.append(f"{np.mean(me)*1000:6.1f}/{np.mean(mx)*1000:5.1f}")
        print(f"{name:>14} | {len(idx):>6} | " + " | ".join([f"{c:>16}" for c in cols]))

    print("\n说明: 每格是该桶的 平均mean误差 / 平均max误差 (mm)。")
    print("若'大弯'桶的误差远高于'直行'桶，说明固定控制点数把大弯在标签里拉直了。")


if __name__ == '__main__':
    main()
