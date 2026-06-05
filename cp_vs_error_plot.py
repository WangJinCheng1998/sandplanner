#!/usr/bin/env python3
"""
画图：控制点数 vs 重建误差(mean / max)，按转弯幅度分桶(直行/中弯/大弯)。
重点展示「大弯误差」随控制点数的变化。基于真实 GT 轨迹段。

Plot: number of control points vs reconstruction error (mean / max),
bucketed by turn magnitude (straight / medium turn / large turn).
Emphasis on how the large-turn error varies with the number of control
points. Based on real GT trajectory segments.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.interpolate import BSpline
import importlib.util

_bp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sand_planner', 'utils', 'bspline.py')
_spec = importlib.util.spec_from_file_location('_bs', _bp)
_bs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_bs)
resample_equal_arclength = _bs.resample_equal_arclength
_chord = _bs._chord_length_param
_solve = _bs._solve_control_points

DATASET_ROOT = os.environ.get('DATASET_ROOT', 'dataset')
TARGET_LEN = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
TOL, STRIDE, MAX_SEG = 0.5, 15, 4000
M_LIST = [4, 6, 8, 10, 12, 14, 16, 20]


def fit_err(seg, m, k=3, rs=64):
    Q = resample_equal_arclength(np.round(seg.astype(float), 2), rs)
    u = _chord(Q)
    P, t, _ = _solve(Q, u, k=k, m=m, knot_strategy='quantile', enforce_endpoints=True, ridge_lambda=1e-6)
    fit = np.stack([BSpline(t, P[:, ax], k, extrapolate=False)(u) for ax in range(3)], -1)
    e = np.linalg.norm(Q - fit, axis=1)
    return float(np.mean(e)), float(np.max(e))


def net_turn_deg(seg):
    d = np.diff(seg[:, :2], axis=0); n = np.linalg.norm(d, axis=1); d = d[n > 1e-6]
    if len(d) < 2: return 0.0
    h0 = np.arctan2(d[0, 1], d[0, 0]); h1 = np.arctan2(d[-1, 1], d[-1, 0])
    return float(abs(np.degrees(np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0)))))


def collect():
    segs = []
    for ds in sorted(os.listdir(DATASET_ROOT)):
        dsp = os.path.join(DATASET_ROOT, ds)
        if not os.path.isdir(dsp): continue
        for r in sorted(os.listdir(dsp)):
            p = os.path.join(dsp, r, 'traj_xyz.npy')
            if not os.path.exists(p): continue
            try: traj = np.load(p).astype(float)
            except Exception: continue
            if traj.ndim != 2 or traj.shape[0] < 8 or traj.shape[1] != 3: continue
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))])
            n = len(traj)
            for s in range(0, n - 1, STRIDE):
                e = int(np.searchsorted(cum, cum[s] + TARGET_LEN))
                if e >= n: break
                if abs((cum[e] - cum[s]) - TARGET_LEN) <= TOL and (e - s) >= 6:
                    segs.append(traj[s:e + 1].copy())
                    if len(segs) >= MAX_SEG: return segs
    return segs


def main():
    segs = collect()
    turns = np.array([net_turn_deg(s) for s in segs])
    buckets = [('Straight (<20°)', turns < 20, 'tab:green'),
               ('Medium (20-60°)', (turns >= 20) & (turns < 60), 'tab:orange'),
               ('Large turn (>=60°)', turns >= 60, 'tab:red')]
    print(f"{len(segs)} segments @ {TARGET_LEN}m | "
          + ", ".join([f"{n}:{int(m.sum())}" for n, m, _ in buckets]))

    # 计算 mean/max 误差表：res[bucket_name][m] = (mean_mm, max_mm)
    # Compute the mean/max error table: res[bucket_name][m] = (mean_mm, max_mm)
    res = {n: {} for n, _, _ in buckets}
    for name, mask, _ in buckets:
        idx = np.where(mask)[0]
        for m in M_LIST:
            me, mx = [], []
            for i in idx:
                try:
                    a, b = fit_err(segs[i], m); me.append(a); mx.append(b)
                except Exception: pass
            res[name][m] = (np.mean(me) * 1000, np.mean(mx) * 1000)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (axm, axx) = plt.subplots(1, 2, figsize=(14, 6))
    for name, _, color in buckets:
        mean_y = [res[name][m][0] for m in M_LIST]
        max_y = [res[name][m][1] for m in M_LIST]
        axm.plot(M_LIST, mean_y, 'o-', color=color, label=name, linewidth=2)
        axx.plot(M_LIST, max_y, 's-', color=color, label=name, linewidth=2)

    for ax, title in [(axm, 'Mean reconstruction error'), (axx, 'Max reconstruction error')]:
        ax.set_xlabel('number of control points (m)')
        ax.set_ylabel('error (mm)')
        ax.set_title(f'{title}  ({TARGET_LEN}m paths, n={len(segs)})')
        ax.grid(True, alpha=0.3); ax.legend()
        ax.axvline(8, color='gray', ls=':', alpha=0.6)
        ax.axvline(12, color='blue', ls=':', alpha=0.6)
        ax.set_xticks(M_LIST)

    plt.tight_layout()
    out = f'cp_vs_error_{TARGET_LEN}m.png'
    plt.savefig(out, dpi=150)
    print(f"saved: {os.path.abspath(out)}")

    # 大弯桶的数值表 / Numeric table for the large-turn bucket
    print(f"\n=== Large-turn (>=60°) error vs control points @ {TARGET_LEN}m ===")
    print(f"{'m':>4} | {'mean(mm)':>9} | {'max(mm)':>8}")
    for m in M_LIST:
        mm, mx = res['Large turn (>=60°)'][m]
        print(f"{m:>4} | {mm:>9.1f} | {mx:>8.1f}")


if __name__ == '__main__':
    main()
