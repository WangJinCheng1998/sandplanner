"""扫描数据集，按 dataloader 的采样逻辑统计样本弧长（arc length）分布。 / Scan the dataset and compute the per-sample arc length distribution following the dataloader's sampling logic.

复刻 dataloader_bspline.py 的 (start_idx, end_idx) 采样规则，但只计算弧长，
不做相对坐标变换、深度图加载、B-spline 拟合等重活。

Replicates the (start_idx, end_idx) sampling rule from dataloader_bspline.py, but only
computes the arc length; it skips the heavy work such as relative-frame transforms, depth
image loading, and B-spline fitting.
"""

import argparse
import glob
import os
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def discover_runs(dataset_root, min_trajectory_length=5):
    """返回一个 np.ndarray 列表，每个元素是一条 run 的 traj_xyz 轨迹。 / Return a list of np.ndarray, each holding the traj_xyz trajectory of one run."""
    runs = []
    for traj_path in sorted(glob.glob(os.path.join(dataset_root, "*", "*", "traj_xyz.npy"))):
        try:
            traj = np.load(traj_path)
            if len(traj) >= min_trajectory_length:
                runs.append(traj)
        except Exception as e:
            print(f"skip {traj_path}: {e}")
    return runs


def sample_arc_lengths(runs, n_samples, min_gap, max_gap, seed=0):
    rng = random.Random(seed)
    out = []
    n_skipped = 0
    for _ in range(n_samples):
        # 跟 dataloader 一样：随机选一条 run，再随机选起止索引
        # Same as the dataloader: pick a random run, then random start/end indices
        traj = runs[rng.randrange(len(runs))]
        run_length = len(traj)
        max_start_idx = run_length - min(min_gap + 1, max_gap + 1)
        if max_start_idx < 0:
            n_skipped += 1
            continue
        start_idx = rng.randint(0, max_start_idx)
        min_end_idx = start_idx + min_gap
        max_end_idx = min(start_idx + max_gap, run_length - 1)
        if min_end_idx > max_end_idx:
            n_skipped += 1
            continue
        end_idx = rng.randint(min_end_idx, max_end_idx)

        seg = traj[start_idx:end_idx + 1]
        diffs = np.diff(seg, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        arc = float(seg_lengths.sum())
        out.append(arc)
    return np.array(out), n_skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root",
                   default=os.environ.get("DATASET_ROOT", "dataset"))
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--min_gap", type=int, default=2)
    p.add_argument("--max_gap", type=int, default=42)
    p.add_argument("--max_arc_length", type=float, default=3.0,
                   help="标在图上的截断线（不真截断）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="arc_length_dist.png")
    args = p.parse_args()

    print(f"扫描数据集: {args.dataset_root}")
    runs = discover_runs(args.dataset_root)
    print(f"  有效 runs: {len(runs)}")
    print(f"  run 长度统计: min={min(len(r) for r in runs)}, "
          f"median={int(np.median([len(r) for r in runs]))}, "
          f"max={max(len(r) for r in runs)}")

    print(f"\n采样 {args.n_samples} 个 (start, end) 对 "
          f"(min_gap={args.min_gap}, max_gap={args.max_gap}) ...")
    arcs, n_skipped = sample_arc_lengths(
        runs, args.n_samples, args.min_gap, args.max_gap, seed=args.seed
    )
    print(f"  有效样本: {len(arcs)}, 跳过: {n_skipped}")

    # 统计 / Statistics
    cutoff = args.max_arc_length
    pct_above = (arcs > cutoff).mean() * 100
    print(f"\n--- 弧长分布 ---")
    print(f"  mean   = {arcs.mean():.3f} m")
    print(f"  median = {np.median(arcs):.3f} m")
    print(f"  p1     = {np.percentile(arcs, 1):.3f} m")
    print(f"  p25    = {np.percentile(arcs, 25):.3f} m")
    print(f"  p75    = {np.percentile(arcs, 75):.3f} m")
    print(f"  p95    = {np.percentile(arcs, 95):.3f} m")
    print(f"  p99    = {np.percentile(arcs, 99):.3f} m")
    print(f"  max    = {arcs.max():.3f} m")
    print(f"  > {cutoff}m: {pct_above:.1f}%   ← 这些样本会被 max_arc_length 截断")

    # 画图 / Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, max(arcs.max(), cutoff * 2.5), 80)
    ax.hist(arcs, bins=bins, color="#3a7fb8", alpha=0.85, edgecolor="white")
    ax.axvline(cutoff, color="crimson", linestyle="--", linewidth=2,
               label=f"max_arc_length = {cutoff}m ({pct_above:.1f}% above)")
    ax.axvline(arcs.mean(), color="orange", linestyle=":", linewidth=2,
               label=f"mean = {arcs.mean():.2f}m")
    ax.axvline(np.median(arcs), color="green", linestyle=":", linewidth=2,
               label=f"median = {np.median(arcs):.2f}m")
    ax.set_xlabel("Sample arc length (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"Sampled trajectory arc length distribution\n"
                 f"(dataset={os.path.basename(args.dataset_root)}, "
                 f"min_gap={args.min_gap}, max_gap={args.max_gap}, n={len(arcs)})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"\n保存图: {args.out}")


if __name__ == "__main__":
    main()
