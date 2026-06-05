"""
轨迹评估向量化优化 / Vectorized trajectory evaluation.

优化策略：
1. 批量计算 clearance cost（避免逐条轨迹的 for 循环）
2. 批量计算 length cost
3. 批量计算 goal cost
4. 向量化排序与统计

预期加速：2-3 倍。

Optimization strategy:
1. Batched clearance cost (avoids a per-trajectory for loop).
2. Batched length cost.
3. Batched goal cost.
4. Vectorized sorting and statistics.

Expected speedup: 2-3x.
"""

import numpy as np
from typing import List, Dict
import time


def compute_clearance_cost_batch(trajectories: List[np.ndarray], 
                                 esdf_query_batch_fn,
                                 safety_margin: float = 0.15,
                                 max_points: int = None) -> np.ndarray:
    """
    批量计算 clearance cost（向量化版本）/ Batched clearance cost (vectorized).

    Args:
        trajectories: 轨迹列表，[B 条轨迹，每条 (M_i, 3)] / list of trajectories, [B trajectories, each (M_i, 3)].
        esdf_query_batch_fn: 批量 ESDF 查询函数 / batched ESDF query function.
        safety_margin: 安全余量 / safety margin.
        max_points: 最大查询点数 / maximum number of query points.

    Returns:
        clearance_costs: (B,) 每条轨迹的 clearance cost / clearance cost per trajectory.
    """
    B = len(trajectories)
    clearance_costs = np.zeros(B)
    
    # 收集所有需要查询的点 / Collect every point that needs to be queried.
    all_points = []
    point_counts = []

    for traj in trajectories:
        M = len(traj)
        if max_points is not None and max_points < M:
            # 等间隔重采样 / Resample at equal index intervals.
            indices = np.linspace(0, M-1, max_points, dtype=int)
            query_points = traj[indices]
        else:
            query_points = traj
        
        all_points.append(query_points)
        point_counts.append(len(query_points))
    
    # 拼接为一个大数组 / Concatenate into a single large array.
    all_points_concat = np.vstack(all_points)  # (Total_points, 3)

    # 批量 ESDF 查询（一次性查询所有点）/ Batched ESDF query (query all points at once).
    all_distances = esdf_query_batch_fn(all_points_concat)  # (Total_points,)

    # 按轨迹切回各自的距离段 / Split the distances back per trajectory.
    optimal_min = safety_margin
    optimal_max = 0.6
    discount_factor = 0.75
    start_idx = 0
    for i, count in enumerate(point_counts):
        end_idx = start_idx + count
        distances = all_distances[start_idx:end_idx]

        # 计算安全余量惩罚 / Compute the clearance/safety-margin penalty.
        safety_violations = np.maximum(0, optimal_min - distances)
        efficiency_violations = np.maximum(0, distances - optimal_max)
        total_penalties = safety_violations ** 2 + 0.1 * (efficiency_violations ** 2)

        # 折扣权重 / Discount weights.
        discount_weights = np.power(discount_factor, np.arange(count))
        weighted_penalties_sum = np.sum(total_penalties * discount_weights)
        weights_sum = np.sum(discount_weights)
        
        clearance_costs[i] = weighted_penalties_sum / weights_sum if weights_sum > 0 else 0.0
        start_idx = end_idx
    
    return clearance_costs


def compute_length_cost_batch(trajectories: List[np.ndarray]) -> np.ndarray:
    """
    批量计算 length cost（向量化版本）/ Batched length cost (vectorized).

    Args:
        trajectories: 轨迹列表 / list of trajectories.

    Returns:
        length_costs: (B,) 每条轨迹的 length cost / length cost per trajectory.
    """
    length_costs = []

    for traj in trajectories:
        if len(traj) < 2:
            length_costs.append(0.0)
            continue

        # 向量化计算每段长度 / Compute per-segment lengths in a vectorized way.
        segment_lengths = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        total_length = np.sum(segment_lengths)
        length_costs.append(total_length)
    
    return np.array(length_costs)


def compute_goal_cost_batch(trajectories: List[np.ndarray], 
                           goal_point: np.ndarray) -> np.ndarray:
    """
    批量计算 goal cost（向量化版本）/ Batched goal cost (vectorized).

    Args:
        trajectories: 轨迹列表 / list of trajectories.
        goal_point: 目标点 (3,) / goal point (3,).

    Returns:
        goal_costs: (B,) 每条轨迹的 goal cost / goal cost per trajectory.
    """
    # 收集所有轨迹终点 / Collect the end point of every trajectory.
    end_points = np.array([traj[-1] if len(traj) > 0 else np.zeros(3)
                          for traj in trajectories])  # (B, 3)

    # 批量计算到目标点的距离 / Batched distance to the goal point.
    goal_errors = np.linalg.norm(end_points - goal_point, axis=1)  # (B,)
    goal_costs = goal_errors ** 2
    
    return goal_costs


def evaluate_trajectories_vectorized(
    trajectories: List[np.ndarray],
    goal_point: np.ndarray,
    esdf_query_batch_fn,
    weights: Dict[str, float] = {'clear': 10000.0, 'length': 0.01, 'goal': 0.01},
    safety_margin: float = 0.15,
    clearance_max_points: int = 40
) -> Dict:
    """
    向量化批量评估轨迹（优化版本）/ Vectorized batched trajectory evaluation (optimized).

    优化策略：
    1. 批量 ESDF 查询（一次性查询所有轨迹的所有点）
    2. 向量化的 cost 计算
    3. 批量排序

    预期加速：2-3 倍。

    Optimization strategy:
    1. Batched ESDF query (query all points of all trajectories at once).
    2. Vectorized cost computation.
    3. Batched sorting.

    Expected speedup: 2-3x.

    Args:
        trajectories: 轨迹列表 / list of trajectories.
        goal_point: 目标点 / goal point.
        esdf_query_batch_fn: 批量 ESDF 查询函数 / batched ESDF query function.
        weights: 权重字典 / dictionary of cost weights.
        safety_margin: 安全余量 / safety margin.
        clearance_max_points: clearance 计算的最大点数 / maximum number of points used for clearance.

    Returns:
        evaluation_results: 评估结果字典 / dictionary of evaluation results.
    """
    B = len(trajectories)

    # 1. 批量计算各项 cost / Batch-compute each cost term.
    clearance_costs = compute_clearance_cost_batch(
        trajectories, esdf_query_batch_fn, safety_margin, clearance_max_points
    )

    length_costs = compute_length_cost_batch(trajectories)

    goal_costs = compute_goal_cost_batch(trajectories, goal_point)

    # 2. 批量计算总 cost / Batch-compute the total cost.
    total_costs = (
        weights['clear'] * clearance_costs +
        weights['length'] * length_costs +
        weights['goal'] * goal_costs
    )

    # 3. 批量排序 / Batched sorting.
    sorted_indices = np.argsort(total_costs)

    # 4. 构建结果 / Build the result entries.
    results = []
    for idx in sorted_indices:
        results.append({
            'trajectory_id': int(idx),
            'total_cost': float(total_costs[idx]),
            'clearance_cost': float(clearance_costs[idx]),
            'length_cost': float(length_costs[idx]),
            'goal_cost': float(goal_costs[idx]),
            'weights': weights.copy(),
            'weighted_costs': {
                'clearance': float(weights['clear'] * clearance_costs[idx]),
                'length': float(weights['length'] * length_costs[idx]),
                'goal': float(weights['goal'] * goal_costs[idx])
            }
        })
    
    # 5. 统计信息 / Summary statistics.
    summary = {
        'num_trajectories': B,
        'best_cost': float(np.min(total_costs)),
        'worst_cost': float(np.max(total_costs)),
        'mean_cost': float(np.mean(total_costs)),
        'std_cost': float(np.std(total_costs)),
        'cost_components_stats': {
            'clearance': {'mean': float(np.mean(clearance_costs)), 'std': float(np.std(clearance_costs))},
            'length': {'mean': float(np.mean(length_costs)), 'std': float(np.std(length_costs))},
            'goal': {'mean': float(np.mean(goal_costs)), 'std': float(np.std(goal_costs))}
        }
    }
    
    return {
        'results': results,
        'summary': summary,
        'weights': weights.copy(),
        'safety_margin': safety_margin
    }


def benchmark_vectorized_evaluation():
    """性能基准测试 / Performance benchmark."""
    from trajectory_evaluation import TrajectoryEvaluator
    
    print("=" * 80)
    print("轨迹评估向量化优化测试")
    print("=" * 80)
    
    # 生成测试数据 / Generate test data.
    num_trajectories = 64
    points_per_traj = 50
    goal_point = np.array([5.0, 0.0, 0.0])

    trajectories = []
    for _ in range(num_trajectories):
        # 随机生成轨迹 / Generate a random trajectory.
        traj = np.cumsum(np.random.randn(points_per_traj, 3) * 0.1, axis=0)
        trajectories.append(traj)

    # 创建模拟的 ESDF 查询函数 / Create mock ESDF query functions.
    def mock_esdf_query(point):
        # 单点查询 / Single-point query.
        return np.random.uniform(0.1, 1.0)

    def mock_esdf_query_batch(points):
        # 批量查询：返回随机距离 / Batched query: return random distances.
        points = np.asarray(points)
        if len(points.shape) == 1:
            # 单点 / Single point.
            return np.random.uniform(0.1, 1.0)
        # 多点 / Multiple points.
        return np.random.uniform(0.1, 1.0, len(points))

    # 将 batch 函数附加为属性 / Attach the batch function as an attribute.
    mock_esdf_query.batch = mock_esdf_query_batch

    # 测试配置 / Test configuration.
    weights = {'clear': 10000.0, 'length': 0.01, 'goal': 0.01}
    safety_margin = 0.15
    clearance_max_points = 40
    num_iterations = 50
    
    print(f"\n测试配置:")
    print(f"  轨迹数量: {num_trajectories}")
    print(f"  每条轨迹点数: {points_per_traj}")
    print(f"  Clearance查询点数: {clearance_max_points}")
    print(f"  迭代次数: {num_iterations}")
    
    # 测试原始版本 / Benchmark the original version.
    print("\n" + "-" * 80)
    print("测试原始版本（逐个轨迹评估）...")

    evaluator = TrajectoryEvaluator(
        esdf_query_function=mock_esdf_query,  # 使用单点查询函数 / Use the single-point query function.
        safety_margin=safety_margin,
        weights=weights
    )
    
    times_original = []
    for _ in range(num_iterations):
        start = time.time()
        result1 = evaluator.evaluate_trajectories(
            trajectories, goal_point, 
            return_ranking=True, 
            clearance_max_points=clearance_max_points
        )
        end = time.time()
        times_original.append((end - start) * 1000)
    
    avg_original = np.mean(times_original[5:])
    print(f"  平均耗时: {avg_original:.2f} ms")
    
    # 测试向量化版本 / Benchmark the vectorized version.
    print("\n测试向量化版本（批量评估）...")
    
    times_vectorized = []
    for _ in range(num_iterations):
        start = time.time()
        result2 = evaluate_trajectories_vectorized(
            trajectories, goal_point,
            mock_esdf_query_batch,
            weights, safety_margin, clearance_max_points
        )
        end = time.time()
        times_vectorized.append((end - start) * 1000)
    
    avg_vectorized = np.mean(times_vectorized[5:])
    print(f"  平均耗时: {avg_vectorized:.2f} ms")
    
    # 性能对比 / Performance comparison.
    print("\n" + "=" * 80)
    print("性能对比")
    print("=" * 80)
    
    speedup = avg_original / avg_vectorized
    time_saved = avg_original - avg_vectorized
    
    print(f"\n原始版本:     {avg_original:.2f} ms")
    print(f"向量化版本:   {avg_vectorized:.2f} ms")
    print(f"\n🚀 优化效果:")
    print(f"  加速比:     {speedup:.1f}x")
    print(f"  节省时间:   {time_saved:.2f} ms")
    
    if time_saved > 10:
        print(f"\n✅ 成功优化轨迹评估，节省 {time_saved:.1f}ms！")
    
    # 验证正确性 / Verify correctness.
    print("\n" + "-" * 80)
    print("验证结果一致性...")
    best_id_original = result1['results'][0]['trajectory_id']
    best_id_vectorized = result2['results'][0]['trajectory_id']
    
    print(f"  原始版本最佳轨迹: {best_id_original}")
    print(f"  向量化版本最佳轨迹: {best_id_vectorized}")
    
    if best_id_original == best_id_vectorized:
        print("  ✅ 结果一致！")
    else:
        print("  ⚠️  结果不一致，需要检查实现")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark_vectorized_evaluation()
