#!/usr/bin/env python3
"""
B-spline 轨迹拟合核心模块 - 固定 8 控制点（3D）/
Core B-spline trajectory fitting module - fixed 8 control points (3D).

相对旧版实现的改进点：
- 使用单一参数化 u（弦长参数）和统一的结点向量（knot vector），而非分别对 x/y/z 独立拟合；
- 明确定义为 clamped cubic B-spline（端点重复 k+1 次），确保曲线过首末控制点；
- 严格 8 个控制点：通过线性最小二乘直接解控制点，而不是从 splrep 的系数长度推测；
- 可选端点硬约束：固定 P0=首点、P7=末点，解内部控制点，得到更稳定一致的拟合。

Improvements over the legacy implementation:
- Use a single parametrization u (chord-length parameter) and a unified knot vector,
  instead of fitting x/y/z independently.
- Explicitly defined as a clamped cubic B-spline (endpoints repeated k+1 times),
  so the curve passes through the first and last control points.
- Strictly 8 control points: solve the control points directly via linear
  least squares, rather than guessing from the coefficient length of splrep.
- Optional hard endpoint constraints: fix P0 = first point and P7 = last point,
  then solve the interior control points for a more stable and consistent fit.
"""

import numpy as np
from scipy import interpolate
from scipy.interpolate import BSpline


def _chord_length_param(points: np.ndarray) -> np.ndarray:
    """弦长参数化 u∈[0,1]，形状 (N,)。/ Chord-length parametrization u∈[0,1], shape (N,)."""
    N = len(points)
    if N <= 1:
        return np.zeros(N, dtype=float)
    d = np.linalg.norm(points[1:] - points[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return np.linspace(0.0, 1.0, N)
    return (s / s[-1]).astype(float)


def resample_equal_arclength(points: np.ndarray, num: int) -> np.ndarray:
    """将折线按等弧长重采样为 num 个点。/ Resample a polyline to num points by equal arc length.

    参数 / Args:
      - points: (N,3) 原始点（N>=1）。若 N==1，重复该点；若总长度为 0，同样返回重复点。/
        (N,3) raw points (N>=1). If N==1, repeat that point; if total length is 0, also return repeats.
      - num: 目标点数（>=2 更有意义）。/ Target number of points (>=2 is more meaningful).
    """
    P = np.asarray(points, dtype=float)
    N = len(P)
    if N == 0:
        return np.zeros((num, 3), dtype=float)
    if N == 1:
        return np.repeat(P, num, axis=0)
    # 弧长 / arc length
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.repeat(P[:1], num, axis=0)
    u_src = s / total
    u_dst = np.linspace(0.0, 1.0, int(num))
    out = np.zeros((len(u_dst), 3), dtype=float)
    for ax in range(3):
        out[:, ax] = np.interp(u_dst, u_src, P[:, ax])
    return out


def _uniform_internal_knots(m: int, k: int) -> np.ndarray:
    """返回 m 个控制点、次数 k 的 clamped B-spline 的内部结点（均匀分布）。/
    Return uniformly spaced internal knots for a clamped B-spline with m control points and degree k.

    总结点向量长度为 m + k + 1；首尾各重复 (k+1) 次 0/1；内部结点数量 = m - k - 1。/
    The full knot vector has length m + k + 1; the ends each repeat 0/1 (k+1) times;
    the number of internal knots is m - k - 1.
    """
    n_internal = m - k - 1
    if n_internal <= 0:
        return np.array([], dtype=float)
    # 等距分成 (n_internal+1) 个区间，取去掉端点后的内部点
    # Split into (n_internal+1) equal intervals and take the interior points (excluding endpoints).
    return np.linspace(0.0, 1.0, n_internal + 2)[1:-1]


def _quantile_internal_knots(u: np.ndarray, m: int, k: int) -> np.ndarray:
    """按 u 的分位点生成内部结点，避免不均匀采样导致的振荡。/
    Generate internal knots from the quantiles of u to avoid oscillation caused by non-uniform sampling.
    """
    n_internal = m - k - 1
    if n_internal <= 0:
        return np.array([], dtype=float)
    qs = [(i + 1) / (n_internal + 1) for i in range(n_internal)]
    return np.quantile(u, qs)


def _build_clamped_knot_vector(m: int, k: int, internal: np.ndarray) -> np.ndarray:
    start = np.zeros(k + 1, dtype=float)
    end = np.ones(k + 1, dtype=float)
    return np.concatenate([start, internal.astype(float), end])


def _design_matrix(u: np.ndarray, t: np.ndarray, k: int, m: int) -> np.ndarray:
    """构造 B-spline 设计矩阵 N(u) ∈ R^{N×m}，第 j 列为第 j 个基函数在所有 u 上的取值。/
    Build the B-spline design matrix N(u) ∈ R^{N×m}; column j holds the j-th basis function
    evaluated at all u.

    通过构造单位系数的 BSpline 并批量评估，当 m 很小（8）时开销可接受。/
    Built by constructing a unit-coefficient BSpline and batch-evaluating it; the cost is
    acceptable when m is small (8).
    """
    N = np.zeros((len(u), m), dtype=float)
    for j in range(m):
        c = np.zeros(m)
        c[j] = 1.0
        basis_j = BSpline(t, c, k, extrapolate=False)
        N[:, j] = basis_j(u)
    # 将 NaN（结点范围外）置 0，避免数值污染
    # Set NaN values (outside the knot range) to 0 to avoid numerical contamination.
    N[~np.isfinite(N)] = 0.0
    return N


def _solve_control_points(Q: np.ndarray, u: np.ndarray, k: int = 3, m: int = 8,
                          knot_strategy: str = 'quantile', enforce_endpoints: bool = True,
                          ridge_lambda: float = 1e-6) -> tuple:
    """解 3D 控制点 P ∈ R^{m×3}，使得 N(u)P ≈ Q。/ Solve for 3D control points P ∈ R^{m×3} such that N(u)P ≈ Q.

    参数 / Args:
      - Q: (N,3) 观测点。/ (N,3) observed points.
      - u: (N,) 参数。/ (N,) parameter values.
      - k: 次数（3 表示三次）。/ Degree (3 means cubic).
      - m: 控制点个数（8）。/ Number of control points (8).
      - knot_strategy: 'quantile' 或 'uniform'。/ 'quantile' or 'uniform'.
      - enforce_endpoints: 是否将 P0=Q0, P_{m-1}=Q_{-1}。/ Whether to set P0=Q0 and P_{m-1}=Q_{-1}.
      - ridge_lambda: 岭正则强度，避免病态。/ Ridge regularization strength, to avoid ill-conditioning.

    返回 / Returns:
      (P, t, Nmat) 控制点、结点向量、设计矩阵。/ (P, t, Nmat): control points, knot vector, design matrix.
    """
    assert m >= k + 1, "控制点数必须 >= 次数+1"
    if knot_strategy == 'uniform':
        internal = _uniform_internal_knots(m, k)
    else:
        internal = _quantile_internal_knots(u, m, k)
        if np.any(np.diff(internal) <= 1e-8):
            # 数据极端不均匀时回退为均匀结点
            # Fall back to uniform knots when the data is extremely non-uniform.
            internal = _uniform_internal_knots(m, k)

    t = _build_clamped_knot_vector(m, k, internal)
    Nmat = _design_matrix(u, t, k, m)  # 形状 (N, m) / shape (N, m)

    # 端点硬约束：固定 P0, P_{m-1} / Hard endpoint constraint: fix P0 and P_{m-1}.
    if enforce_endpoints and len(Q) >= 2:
        P0 = Q[0]
        Pn = Q[-1]
        fixed_cols = [0, m - 1]
        free_cols = [j for j in range(m) if j not in fixed_cols]

        Nf = Nmat[:, fixed_cols]          # 形状 (N,2) / shape (N,2)
        Ni = Nmat[:, free_cols]           # 形状 (N,m-2) / shape (N,m-2)
        rhs = Q - Nf @ np.vstack([P0, Pn])  # 形状 (N,3) / shape (N,3)

        # 岭回归 Ni Pi = rhs / Ridge regression Ni Pi = rhs.
        A = Ni
        # 正则矩阵 / regularization matrix
        reg = np.sqrt(ridge_lambda) * np.eye(Ni.shape[1])
        # 按轴分别求解 / solve each axis separately
        Pi = np.zeros((Ni.shape[1], 3))
        for ax in range(3):
            b = rhs[:, ax]
            A_aug = np.vstack([A, reg])
            b_aug = np.concatenate([b, np.zeros(reg.shape[0])])
            sol, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
            Pi[:, ax] = sol

        P = np.zeros((m, 3))
        P[0] = P0
        P[-1] = Pn
        P[free_cols] = Pi
    else:
        # 无端点硬约束的最小二乘（不推荐）
        # Least squares without hard endpoint constraints (not recommended).
        A = Nmat
        reg = np.sqrt(ridge_lambda) * np.eye(m)
        P = np.zeros((m, 3))
        for ax in range(3):
            b = Q[:, ax]
            A_aug = np.vstack([A, reg])
            b_aug = np.concatenate([b, np.zeros(reg.shape[0])])
            sol, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
            P[:, ax] = sol

    return P, t, Nmat


def fit_trajectory_8cp_clamped(
    trajectory_points: np.ndarray,
    degree: int = 3,
    enforce_endpoints: bool = True,
    knot_strategy: str = 'quantile',
    ridge_lambda: float = 1e-3,
    num_samples: int = 100,
    smoothing_lambda: float | None = None,
    resample_points: int = 30,
    num_control_points: int = 8,
) -> dict:
    """使用严格 8 控制点的 clamped cubic B-spline 拟合 3D 轨迹。/
    Fit a 3D trajectory with a clamped cubic B-spline using strictly 8 control points.

    预处理：在拟合前对原始轨迹坐标四舍五入保留两位小数，然后执行等弧长重采样到固定点数，
    再求解控制点。/
    Preprocessing: round the raw trajectory coordinates to two decimals before fitting,
    then resample to a fixed number of points by equal arc length, and finally solve the control points.

    参数 / Args:
      - trajectory_points: (N,3) 观测点（任意 N>=1）。/ (N,3) observed points (any N>=1).
      - degree: 次数（默认 3）。/ Degree (default 3).
      - enforce_endpoints: 是否强制曲线通过起止点。/ Whether to force the curve through the start/end points.
      - knot_strategy: 'quantile'（推荐）或 'uniform'。/ 'quantile' (recommended) or 'uniform'.
      - ridge_lambda: 岭正则强度。/ Ridge regularization strength.
      - num_samples: 用于可视化/评估的稠密采样点数。/ Number of dense samples for visualization/evaluation.
      - resample_points: 等弧长重采样的目标点数（默认 30）。/
        Target number of points for equal-arc-length resampling (default 30).

    返回 / Returns:
      dict，包含以下字段 / dict containing:
      - fitted_points: (num_samples,3) 拟合曲线点。/ (num_samples,3) fitted curve points.
      - control_points: (8,3)。/ (8,3) control points.
      - knots: (m+k+1,) 结点向量。/ (m+k+1,) knot vector.
      - max_error, mean_error: 最大误差、平均误差。/ maximum error, mean error.
      - success: bool 是否成功。/ bool, whether the fit succeeded.
    """
    m = int(num_control_points)
    k = int(degree)
    if trajectory_points is None or len(trajectory_points) < 2 or trajectory_points.shape[1] != 3:
        return {
            'fitted_points': trajectory_points if isinstance(trajectory_points, np.ndarray) else None,
            'max_error': 0.0,
            'mean_error': 0.0,
            'success': False,
            'control_points_count': m,
            'error_msg': '输入轨迹无效或点数不足'
        }

    # 原始轨迹坐标四舍五入保留两位小数
    # Round the raw trajectory coordinates to two decimals.
    Q_raw = np.asarray(trajectory_points, dtype=float)
    Q_raw = np.round(Q_raw, 2)
    if smoothing_lambda is not None:
        # 兼容旧脚本参数名：将 smoothing_lambda 作为岭正则强度
        # Backward-compat with the old script parameter name: use smoothing_lambda as the ridge strength.
        ridge_lambda = float(smoothing_lambda)
    # 等弧长预重采样：始终执行到固定点数
    # Equal-arc-length pre-resampling: always run to a fixed number of points.
    try:
        Q = resample_equal_arclength(Q_raw, int(resample_points))
    except Exception:
        Q = Q_raw

    try:
        u = _chord_length_param(Q)
        P, t, Nmat = _solve_control_points(
            Q, u, k=k, m=m,
            knot_strategy=knot_strategy,
            enforce_endpoints=enforce_endpoints,
            ridge_lambda=ridge_lambda,
        )

        # 使用统一的 knots 和控制点，分别构造 3 个坐标轴的样条
        # Build a spline for each of the 3 axes using the unified knots and control points.
        bs_x = BSpline(t, P[:, 0], k, extrapolate=False)
        bs_y = BSpline(t, P[:, 1], k, extrapolate=False)
        bs_z = BSpline(t, P[:, 2], k, extrapolate=False)

        u_dense = np.linspace(0.0, 1.0, int(num_samples))
        X = bs_x(u_dense)
        Y = bs_y(u_dense)
        Z = bs_z(u_dense)
        fitted = np.stack([X, Y, Z], axis=-1)

        # 误差：在原始 u 上评估 / Error: evaluate at the original u values.
        Xo = bs_x(u)
        Yo = bs_y(u)
        Zo = bs_z(u)
        err = np.linalg.norm(Q - np.stack([Xo, Yo, Zo], axis=-1), axis=1)
        max_err = float(np.max(err))
        mean_err = float(np.mean(err))

        return {
            'fitted_points': fitted,
            'max_error': max_err,
            'mean_error': mean_err,
            'success': True,
            'control_points_count': m,
            'control_points': P.astype(float),
            'knots': t.astype(float),
            'degree': k,
        }
    except Exception as e:
        # 最后回退：线性重采样 + 8 控制点的线性采样
        # Last-resort fallback: linear resampling plus linear sampling of 8 control points.
        try:
            t_src = np.linspace(0.0, 1.0, len(Q))
            t_dst = np.linspace(0.0, 1.0, num_samples)
            fitted = np.stack([
                np.interp(t_dst, t_src, Q[:, 0]),
                np.interp(t_dst, t_src, Q[:, 1]),
                np.interp(t_dst, t_src, Q[:, 2])
            ], axis=-1)
            # 8 个控制点（按轨迹线性采样）/ 8 control points (linearly sampled along the trajectory).
            t_cp = np.linspace(0.0, 1.0, m)
            cp = np.stack([
                np.interp(t_cp, t_src, Q[:, 0]),
                np.interp(t_cp, t_src, Q[:, 1]),
                np.interp(t_cp, t_src, Q[:, 2])
            ], axis=-1)
        except Exception:
            fitted = Q.copy()
            cp = np.tile(Q[0], (m, 1)) if len(Q) > 0 else np.zeros((m, 3))

        return {
            'fitted_points': fitted,
            'max_error': 0.0,
            'mean_error': 0.0,
            'success': False,
            'control_points_count': m,
            'control_points': cp.astype(float),
            'error_msg': f'B样条拟合失败: {str(e)}'
        }


def fit_trajectory_8cp(trajectory_points,
                       degree: int = 3,
                       enforce_endpoints: bool = True,
                       return_control_points: bool = False,
                       num_control_points: int = 8):
    """向后兼容接口：代理到严格的 clamped B-spline 控制点实现。/
    Backward-compatible interface: delegates to the strict clamped B-spline control-point implementation.

    注意：返回 num_control_points 个控制点（字段 'control_points'）；默认 8 与旧接口兼容。/
    Note: returns num_control_points control points (field 'control_points'); the default 8 is
    compatible with the legacy interface.
    """
    res = fit_trajectory_8cp_clamped(
        np.asarray(trajectory_points),
        degree=int(degree),
        enforce_endpoints=bool(enforce_endpoints),
        knot_strategy='quantile',
        ridge_lambda=1e-6,
        num_samples=100,
        num_control_points=num_control_points,
    )
    if not return_control_points:
        # 保留字段但不强制访问 / Keep the fields but do not force access.
        res = {k: v for k, v in res.items() if k != 'control_points'} | {
            'control_points_count': num_control_points
        }
    return res


def batch_fit_trajectories_8cp(trajectory_list, return_control_points=False):
    """
    批量对多条轨迹做 8 控制点 B-spline 拟合。/
    Batch-fit multiple trajectories with an 8-control-point B-spline.

    Args:
        trajectory_list: numpy 数组列表，每个数组都是 (N, 3) 的轨迹点。/
            list of numpy arrays, each being (N, 3) trajectory points.
        return_control_points: bool，是否返回控制点坐标。/ bool, whether to return control-point coordinates.

    Returns:
        list: 每条轨迹的拟合结果字典列表。/ list of fitting-result dicts, one per trajectory.
    """
    
    results = []
    
    for i, trajectory in enumerate(trajectory_list):
        print(f"处理轨迹 {i+1}/{len(trajectory_list)}: {len(trajectory)}个点")
        result = fit_trajectory_8cp_clamped(trajectory)

        if result['success']:
            print(f"  ✅ 拟合成功 - 最大误差: {result['max_error']:.6f}m, 平均误差: {result['mean_error']:.6f}m")
            if return_control_points and 'control_points' in result:
                print(f"     控制点数量: {len(result['control_points'])}")
        else:
            print(f"  ❌ 拟合失败 - {result.get('error_msg', '未知错误')}")

        results.append(result)
    
    return results


def visualize_control_points(trajectory_points, result, title=None):
    """
    可视化 B-spline 轨迹和控制点。/ Visualize the B-spline trajectory and its control points.

    Args:
        trajectory_points: numpy array，原始轨迹点。/ numpy array, the raw trajectory points.
        result: fit_trajectory_8cp 返回的结果（需要包含控制点）。/
            result returned by fit_trajectory_8cp (must contain control points).
        title: 图片标题。/ figure title.

    Returns:
        matplotlib figure 对象。/ a matplotlib figure object.
    """
    
    if not result.get('success', False) or 'control_points' not in result:
        print("❌ 无法可视化：拟合失败或未包含控制点信息")
        return None
    
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from mpl_toolkits.mplot3d import Axes3D
    
    fitted_points = result['fitted_points']
    control_points = result['control_points']
    
    fig = plt.figure(figsize=(16, 12))

    # 辅助函数：将坐标范围对齐到 0.5 的倍数，并将主刻度设为 0.5 m
    # Helper: align coordinate limits to multiples of 0.5 and set the major tick to 0.5 m.
    def _align_limits(vmin: float, vmax: float, step: float = 0.5):
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = 0.0
        if vmin == vmax:
            vmin -= step
            vmax += step
        lo = np.floor(vmin / step) * step
        hi = np.ceil(vmax / step) * step
        if hi - lo < step:
            hi = lo + step
        return float(lo), float(hi)

    def _set_2d_axis_ticks_equal(ax, xs, ys, step: float = 0.5):
        xmin, xmax = _align_limits(np.nanmin(xs), np.nanmax(xs), step)
        ymin, ymax = _align_limits(np.nanmin(ys), np.nanmax(ys), step)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_locator(MultipleLocator(step))
        try:
            ax.set_aspect('equal', adjustable='box')
        except Exception:
            ax.axis('equal')

    def _set_3d_axis_ticks(ax, xs, ys, zs, step: float = 0.5):
        xmin, xmax = _align_limits(np.nanmin(xs), np.nanmax(xs), step)
        ymin, ymax = _align_limits(np.nanmin(ys), np.nanmax(ys), step)
        zmin, zmax = _align_limits(np.nanmin(zs), np.nanmax(zs), step)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xticks(np.arange(xmin, xmax + 1e-9, step))
        ax.set_yticks(np.arange(ymin, ymax + 1e-9, step))
        ax.set_zticks(np.arange(zmin, zmax + 1e-9, step))
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    
    if title:
        fig.suptitle(f'{title} - B样条拟合与控制点', fontsize=16)
    else:
        fig.suptitle('B样条拟合与控制点可视化', fontsize=16)
    
    # 创建 2x2 的子图布局 / Create a 2x2 subplot layout.
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # XY 投影 / XY projection
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xy.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
               'ko-', linewidth=2, markersize=4, label='原始轨迹', alpha=0.7)
    ax_xy.plot(fitted_points[:, 0], fitted_points[:, 1], 
               'b-', linewidth=2, label='B样条拟合')
    ax_xy.scatter(control_points[:, 0], control_points[:, 1], 
                  color='red', s=60, marker='^', label='控制点', zorder=5)
    ax_xy.scatter(trajectory_points[0, 0], trajectory_points[0, 1], 
                  color='green', s=80, marker='o', label='起点', zorder=5)
    ax_xy.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], 
                  color='red', s=80, marker='s', label='终点', zorder=5)
    ax_xy.set_title('XY投影')
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)
    _set_2d_axis_ticks_equal(
        ax_xy,
        np.concatenate([trajectory_points[:, 0], fitted_points[:, 0], control_points[:, 0]]),
        np.concatenate([trajectory_points[:, 1], fitted_points[:, 1], control_points[:, 1]]),
        step=0.5,
    )
    
    # XZ 投影 / XZ projection
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_xz.plot(trajectory_points[:, 0], trajectory_points[:, 2], 
               'ko-', linewidth=2, markersize=4, label='原始轨迹', alpha=0.7)
    ax_xz.plot(fitted_points[:, 0], fitted_points[:, 2], 
               'b-', linewidth=2, label='B样条拟合')
    ax_xz.scatter(control_points[:, 0], control_points[:, 2], 
                  color='red', s=60, marker='^', label='控制点', zorder=5)
    ax_xz.scatter(trajectory_points[0, 0], trajectory_points[0, 2], 
                  color='green', s=80, marker='o', label='起点', zorder=5)
    ax_xz.scatter(trajectory_points[-1, 0], trajectory_points[-1, 2], 
                  color='red', s=80, marker='s', label='终点', zorder=5)
    ax_xz.set_title('XZ投影')
    ax_xz.set_xlabel('X (m)')
    ax_xz.set_ylabel('Z (m)')
    ax_xz.legend()
    ax_xz.grid(True, alpha=0.3)
    _set_2d_axis_ticks_equal(
        ax_xz,
        np.concatenate([trajectory_points[:, 0], fitted_points[:, 0], control_points[:, 0]]),
        np.concatenate([trajectory_points[:, 2], fitted_points[:, 2], control_points[:, 2]]),
        step=0.5,
    )
    
    # YZ 投影 / YZ projection
    ax_yz = fig.add_subplot(gs[1, 0])
    ax_yz.plot(trajectory_points[:, 1], trajectory_points[:, 2], 
               'ko-', linewidth=2, markersize=4, label='原始轨迹', alpha=0.7)
    ax_yz.plot(fitted_points[:, 1], fitted_points[:, 2], 
               'b-', linewidth=2, label='B样条拟合')
    ax_yz.scatter(control_points[:, 1], control_points[:, 2], 
                  color='red', s=60, marker='^', label='控制点', zorder=5)
    ax_yz.scatter(trajectory_points[0, 1], trajectory_points[0, 2], 
                  color='green', s=80, marker='o', label='起点', zorder=5)
    ax_yz.scatter(trajectory_points[-1, 1], trajectory_points[-1, 2], 
                  color='red', s=80, marker='s', label='终点', zorder=5)
    ax_yz.set_title('YZ投影')
    ax_yz.set_xlabel('Y (m)')
    ax_yz.set_ylabel('Z (m)')
    ax_yz.legend()
    ax_yz.grid(True, alpha=0.3)
    _set_2d_axis_ticks_equal(
        ax_yz,
        np.concatenate([trajectory_points[:, 1], fitted_points[:, 1], control_points[:, 1]]),
        np.concatenate([trajectory_points[:, 2], fitted_points[:, 2], control_points[:, 2]]),
        step=0.5,
    )
    
    # 3D 视图 / 3D view
    ax_3d = fig.add_subplot(gs[1, 1], projection='3d')
    ax_3d.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 
               'ko-', linewidth=2, markersize=4, label='原始轨迹', alpha=0.7)
    ax_3d.plot(fitted_points[:, 0], fitted_points[:, 1], fitted_points[:, 2], 
               'b-', linewidth=2, label='B样条拟合')
    ax_3d.scatter(control_points[:, 0], control_points[:, 1], control_points[:, 2],
                  color='red', s=60, marker='^', label='控制点', zorder=5)
    ax_3d.scatter(trajectory_points[0, 0], trajectory_points[0, 1], trajectory_points[0, 2], 
                  color='green', s=80, marker='o', label='起点', zorder=5)
    ax_3d.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], trajectory_points[-1, 2], 
                  color='red', s=80, marker='s', label='终点', zorder=5)
    
    # 设置 3D 视图 / Configure the 3D view.
    ax_3d.view_init(elev=20, azim=45)
    ax_3d.set_title(f'3D视图\n最大误差: {result["max_error"]:.6f}m\n平均误差: {result["mean_error"]:.6f}m')
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.legend()
    ax_3d.grid(True, alpha=0.3)
    _set_3d_axis_ticks(
        ax_3d,
        np.concatenate([trajectory_points[:, 0], fitted_points[:, 0], control_points[:, 0]]),
        np.concatenate([trajectory_points[:, 1], fitted_points[:, 1], control_points[:, 1]]),
        np.concatenate([trajectory_points[:, 2], fitted_points[:, 2], control_points[:, 2]]),
        step=0.5,
    )
    
    return fig


def evaluate_fitting_quality(result):
    """
    评估 8 控制点 B-spline 拟合的质量。/ Evaluate the quality of the 8-control-point B-spline fit.

    Args:
        result: fit_trajectory_8cp 返回的结果字典。/ the result dict returned by fit_trajectory_8cp.

    Returns:
        str: 拟合质量评级 ('优秀', '良好', '一般', '较差')。/
            fit-quality rating ('优秀' excellent, '良好' good, '一般' fair, '较差' poor).
    """
    
    if not result['success']:
        return '失败'
    
    mean_error = result['mean_error']
    max_error = result['max_error']
    
    # 基于误差大小的质量评级 / Quality rating based on error magnitude.
    if mean_error < 0.001 and max_error < 0.005:
        return '优秀'
    elif mean_error < 0.003 and max_error < 0.015:
        return '良好'
    elif mean_error < 0.008 and max_error < 0.030:
        return '一般'
    else:
        return '较差'


def visualize_bspline_8cp_clamped(trajectory_points: np.ndarray, result: dict,
                                  title: str | None = None, save_path: str | None = None):
    """与 compare_bspline_batch.py 期望的可视化接口保持一致。/
    Matches the visualization interface expected by compare_bspline_batch.py.
    """
    fig = visualize_control_points(trajectory_points, result, title)
    if fig is not None and save_path:
        import matplotlib.pyplot as plt
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def simple_example():
    """
    简单的使用示例，包含控制点可视化。/ A simple usage example, including control-point visualization.
    """
    
    print("🎯 B样条拟合核心模块示例 - 8个控制点（含控制点输出）")
    print("=" * 60)
    
    # 创建一个简单的 3D 轨迹示例 / Create a simple 3D trajectory example.
    t = np.linspace(0, 2*np.pi, 20)
    x = np.cos(t) * 5
    y = np.sin(t) * 3
    z = t * 0.1  # 轻微的 Z 轴变化 / slight variation along the Z axis
    
    trajectory = np.column_stack([x, y, z])
    
    print(f"📊 示例轨迹: {len(trajectory)}个点")
    print(f"   轨迹范围: X[{x.min():.2f}, {x.max():.2f}], Y[{y.min():.2f}, {y.max():.2f}], Z[{z.min():.2f}, {z.max():.2f}]")
    
    # 进行 8 控制点拟合（不返回控制点）/ Run the 8-control-point fit (without returning control points).
    result_basic = fit_trajectory_8cp(trajectory)

    # 进行 8 控制点拟合（返回控制点）/ Run the 8-control-point fit (returning control points).
    result_with_cp = fit_trajectory_8cp(trajectory, return_control_points=True)

    # 评估拟合质量 / Evaluate the fit quality.
    quality = evaluate_fitting_quality(result_with_cp)
    
    print(f"\n📈 拟合结果:")
    print(f"   成功: {result_with_cp['success']}")
    print(f"   控制点数: {result_with_cp['control_points_count']}")
    print(f"   最大误差: {result_with_cp['max_error']:.6f}m")
    print(f"   平均误差: {result_with_cp['mean_error']:.6f}m")
    print(f"   质量评级: {quality}")
    
    if 'z_is_constant' in result_with_cp:
        print(f"   Z轴处理: {'线性插值' if result_with_cp['z_is_constant'] else 'B样条拟合'}")
    
    # 显示控制点信息 / Print the control-point information.
    if 'control_points' in result_with_cp:
        control_points = result_with_cp['control_points']
        print(f"\n🎯 控制点信息:")
        print(f"   实际控制点数量: {len(control_points)}")
        print(f"   控制点坐标:")
        for i, cp in enumerate(control_points):
            print(f"     CP{i+1}: [{cp[0]:.3f}, {cp[1]:.3f}, {cp[2]:.3f}]")
        
        # 可视化控制点 / Visualize the control points.
        print(f"\n📊 生成控制点可视化图...")
        fig = visualize_control_points(trajectory, result_with_cp, "示例轨迹")
        if fig:
            import matplotlib.pyplot as plt
            plt.savefig('bspline_control_points_example.png', dpi=300, bbox_inches='tight')
            print(f"💾 保存控制点可视化图: bspline_control_points_example.png")
            plt.close(fig)
    
    print(f"\n💡 核心函数使用方法:")
    print(f"   # 基本拟合")
    print(f"   result = fit_trajectory_8cp(trajectory_points)")
    print(f"   ")
    print(f"   # 包含控制点的拟合")
    print(f"   result = fit_trajectory_8cp(trajectory_points, return_control_points=True)")
    print(f"   control_points = result['control_points']")
    print(f"   ")
    print(f"   # 可视化控制点")
    print(f"   fig = visualize_control_points(trajectory_points, result)")
    
    return result_with_cp


def reconstruct_trajectory_from_8cp(control_points: np.ndarray, 
                                   num_points: int = 100,
                                   degree: int = 3) -> np.ndarray:
    """
    从 8 个 B-spline 控制点重建轨迹。/ Reconstruct a trajectory from 8 B-spline control points.

    与 dataloader 中 fit_trajectory_8cp 使用相同的 B-spline 参数设置，确保推理时的轨迹重建与训练时一致。/
    Uses the same B-spline parameter settings as fit_trajectory_8cp in the dataloader, so that
    trajectory reconstruction at inference time matches training.

    Args:
        control_points: (8, 3) 数组，8 个 B-spline 控制点。/ (8, 3) array, 8 B-spline control points.
        num_points: 重建轨迹的点数，默认 100。/ number of points in the reconstructed trajectory, default 100.
        degree: B-spline 次数，默认 3（三次 B-spline）。/ B-spline degree, default 3 (cubic B-spline).

    Returns:
        trajectory: (num_points, 3) 数组，重建的轨迹点。/ (num_points, 3) array, the reconstructed trajectory points.
    """
    from scipy.interpolate import BSpline
    
    # 确保输入格式正确 / Make sure the input has the correct shape.
    control_points = np.asarray(control_points)
    if control_points.ndim != 2 or control_points.shape[1] != 3:
        raise ValueError(f"控制点应该是(m, 3)形状，实际是{control_points.shape}")

    # 创建 clamped knot 向量（与 bspline_core 中一致）
    # 对于 m 个控制点和三次 B-spline，knot 向量长度应为 m + 3 + 1
    # Create a clamped knot vector (consistent with bspline_core).
    # For m control points and a cubic B-spline, the knot vector length should be m + 3 + 1.
    k = degree
    m = control_points.shape[0]  # 控制点数量（从输入自动推断）/ number of control points (inferred from input)

    # Clamped knot vector：起始和结束各重复 k+1 次；
    # 中间段使用 quantile 策略分布（与 fit_trajectory_8cp_clamped 一致）。
    # Clamped knot vector: ends each repeated k+1 times;
    # the middle segment uses the quantile distribution strategy (consistent with fit_trajectory_8cp_clamped).
    inner_knots = np.linspace(0, 1, m - k + 1)
    knots = np.concatenate([
        np.zeros(k),        # 起始重复 k 次 / start repeated k times
        inner_knots,        # 中间部分 / middle section
        np.ones(k)          # 结束重复 k 次 / end repeated k times
    ])
    
    try:
        # 为每个坐标轴创建 B-spline / Create a B-spline for each axis.
        bs_x = BSpline(knots, control_points[:, 0], k, extrapolate=False)
        bs_y = BSpline(knots, control_points[:, 1], k, extrapolate=False)
        bs_z = BSpline(knots, control_points[:, 2], k, extrapolate=False)

        # 在 [0,1] 上均匀采样 / Sample uniformly over [0,1].
        u = np.linspace(0.0, 1.0, num_points)

        # 评估每个坐标 / Evaluate each coordinate.
        x = bs_x(u)
        y = bs_y(u)
        z = bs_z(u)

        # 组合成轨迹 / Assemble into a trajectory.
        trajectory = np.column_stack([x, y, z])
        
        return trajectory.astype(np.float32)
        
    except Exception as e:
        print(f"B样条重建失败: {e}")
        # 如果失败，使用线性插值作为回退 / On failure, fall back to linear interpolation.
        t_control = np.linspace(0, 1, control_points.shape[0])
        t_trajectory = np.linspace(0, 1, num_points)
        
        trajectory = np.zeros((num_points, 3))
        for axis in range(3):
            trajectory[:, axis] = np.interp(t_trajectory, t_control, control_points[:, axis])
        
        return trajectory.astype(np.float32)


def batch_reconstruct_trajectories_from_8cp(control_points_batch: np.ndarray,
                                           num_points: int = 100) -> np.ndarray:
    """
    批量从 8 个控制点重建轨迹。/ Batch-reconstruct trajectories from 8 control points each.

    Args:
        control_points_batch: (batch_size, 8, 3) 批量控制点。/ (batch_size, 8, 3) batch of control points.
        num_points: 每条轨迹的点数。/ number of points per trajectory.

    Returns:
        trajectories: (batch_size, num_points, 3) 批量轨迹。/ (batch_size, num_points, 3) batch of trajectories.
    """
    batch_size = control_points_batch.shape[0]
    trajectories = np.zeros((batch_size, num_points, 3), dtype=np.float32)
    
    for i in range(batch_size):
        trajectories[i] = reconstruct_trajectory_from_8cp(
            control_points_batch[i], num_points
        )
    
    return trajectories


def compute_bspline_arc_length(control_points: np.ndarray = None, 
                              fitted_points: np.ndarray = None,
                              degree: int = 3, 
                              num_samples: int = 1000) -> float:
    """
    计算 B-spline 轨迹的弧长总长度。/ Compute the total arc length of a B-spline trajectory.

    支持两种输入方式 / Supports two input modes:
    1. 从 8 个控制点重建 B-spline 曲线并计算弧长。/ Reconstruct the B-spline curve from 8 control points and compute its arc length.
    2. 直接从已拟合的轨迹点计算弧长。/ Compute the arc length directly from already-fitted trajectory points.

    Args:
        control_points: (8, 3) 数组，8 个 B-spline 控制点，可选。/ (8, 3) array, 8 B-spline control points, optional.
        fitted_points: (N, 3) 数组，已拟合的轨迹点，可选。/ (N, 3) array, already-fitted trajectory points, optional.
        degree: B-spline 次数，默认 3（三次 B-spline）。/ B-spline degree, default 3 (cubic B-spline).
        num_samples: 用于弧长计算的密集采样点数，默认 1000。/ number of dense samples used for arc-length computation, default 1000.

    Returns:
        arc_length: float，B-spline 轨迹的弧长总长度（米）。/ float, total arc length of the B-spline trajectory (meters).

    Note:
        - control_points 和 fitted_points 至少需要提供一个。/ at least one of control_points and fitted_points must be provided.
        - 如果两者都提供，优先使用 fitted_points。/ if both are provided, fitted_points takes precedence.
        - 弧长计算使用数值积分方法，通过密集采样点累加相邻点间距离。/
          The arc length is computed numerically by summing distances between adjacent dense samples.
    """
    # 参数检查 / Argument checks.
    if control_points is None and fitted_points is None:
        raise ValueError("control_points 和 fitted_points 至少需要提供一个")

    # 优先使用已拟合的轨迹点 / Prefer the already-fitted trajectory points.
    if fitted_points is not None:
        trajectory_points = np.asarray(fitted_points)
        if len(trajectory_points) < 2:
            return 0.0
    else:
        # 从控制点重建轨迹 / Reconstruct the trajectory from control points.
        try:
            trajectory_points = reconstruct_trajectory_from_8cp(
                control_points, num_points=num_samples
            )
        except Exception as e:
            print(f"从控制点重建轨迹失败: {e}")
            return 0.0
    
    # 计算弧长：累加相邻点间的距离 / Compute arc length: sum distances between adjacent points.
    try:
        # 计算相邻点之间的距离 / Compute distances between adjacent points.
        distances = np.linalg.norm(trajectory_points[1:] - trajectory_points[:-1], axis=1)

        # 弧长总长度 / total arc length
        arc_length = float(np.sum(distances))
        
        return arc_length
        
    except Exception as e:
        print(f"弧长计算失败: {e}")
        return 0.0


def compute_bspline_arc_length_from_result(fitting_result: dict, 
                                          num_samples: int = 1000) -> float:
    """
    从 B-spline 拟合结果计算弧长总长度。/ Compute the total arc length from a B-spline fitting result.

    这是一个便捷函数，用于直接从 fit_trajectory_8cp 的返回结果计算弧长。/
    A convenience function that computes the arc length directly from the result returned by fit_trajectory_8cp.

    Args:
        fitting_result: fit_trajectory_8cp 返回的结果字典。/ the result dict returned by fit_trajectory_8cp.
        num_samples: 用于弧长计算的密集采样点数，默认 1000。/ number of dense samples used for arc-length computation, default 1000.

    Returns:
        arc_length: float，B-spline 轨迹的弧长总长度（米）。/ float, total arc length of the B-spline trajectory (meters).
    """
    if not fitting_result.get('success', False):
        return 0.0
    
    # 优先使用 fitted_points / Prefer fitted_points.
    if 'fitted_points' in fitting_result:
        return compute_bspline_arc_length(
            fitted_points=fitting_result['fitted_points']
        )

    # 回退到使用控制点 / Fall back to using the control points.
    if 'control_points' in fitting_result:
        return compute_bspline_arc_length(
            control_points=fitting_result['control_points'],
            num_samples=num_samples
        )
    
    return 0.0


if __name__ == "__main__":
    # 运行示例 / Run the example.
    simple_example()
