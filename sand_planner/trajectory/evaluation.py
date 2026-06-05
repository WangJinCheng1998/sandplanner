#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹评估模块 / Trajectory evaluation module.

计算候选轨迹的多项代价：清距/安全余量惩罚、轨迹长度、终点误差。
Computes multiple cost terms for candidate trajectories: clearance/safety-margin
penalty, trajectory length, and goal endpoint error.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union, Callable

# 设置 matplotlib 使用非交互式后端，避免 GUI 相关的错误
# Set matplotlib to a non-interactive backend to avoid GUI-related errors
import matplotlib
# 使用 Anti-Grain Geometry 后端，无需 X11 或其他 GUI
# Use the Anti-Grain Geometry backend; no X11 or other GUI required
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TrajectoryEvaluator:
    """
    轨迹评估器 - 计算轨迹的多项代价 / Trajectory evaluator computing multiple trajectory costs.

    代价函数：
    1. 清距/安全余量惩罚（最关键）：E_clear = (1/M) * Σ ψ(d_i), ψ(d) = ReLU(m-d)²
    2. 轨迹长度：E_len = (1/(M-1)) * Σ ||p_{i+1} - p_i||
    3. 终点误差：E_goal = ||p_M - g||²

    总代价：E_total = w_clear * E_clear + w_len * E_len + w_goal * E_goal

    Cost functions:
    1. Clearance/safety-margin penalty (most critical): E_clear = (1/M) * Σ ψ(d_i), ψ(d) = ReLU(m-d)²
    2. Trajectory length: E_len = (1/(M-1)) * Σ ||p_{i+1} - p_i||
    3. Goal endpoint error: E_goal = ||p_M - g||²

    Total cost: E_total = w_clear * E_clear + w_len * E_len + w_goal * E_goal
    """
    
    def __init__(
        self,
        esdf_query_function: Callable[[np.ndarray], float],
        safety_margin: float = 0.25,  # 安全余量 m（米）/ safety margin m (meters)
        weights: Optional[Dict[str, float]] = None,
        normalize_costs: bool = True
    ):
        """
        初始化轨迹评估器 / Initialize the trajectory evaluator.

        Args:
            esdf_query_function: ESDF 查询函数，输入 3D 点 (x, y, z)，返回距离值 /
                ESDF query function taking a 3D point (x, y, z) and returning the distance value.
            safety_margin: 安全余量 m（米）/ safety margin m (meters).
            weights: 各项代价权重 {'clear': w1, 'length': w2, 'goal': w3} /
                per-term cost weights {'clear': w1, 'length': w2, 'goal': w3}.
            normalize_costs: 是否归一化各项代价 / whether to normalize each cost term.
        """
        self.esdf_query = esdf_query_function
        self.safety_margin = safety_margin
        
        # 默认权重配置 / default weight configuration
        if weights is None:
            weights = {
                'clear': 10.0,    # 清距/安全余量惩罚权重最高 / clearance/safety-margin penalty has the highest weight
                'length': 1.0,    # 轨迹长度权重 / trajectory length weight
                'goal': 1.0       # 终点误差权重 / goal endpoint error weight
            }

        self.weights = weights
        self.normalize_costs = normalize_costs

        print(f"轨迹评估器初始化:")
        print(f"  安全净距: {safety_margin:.2f}m")
        print(f"  权重配置: {weights}")
        print(f"  归一化: {normalize_costs}")
    
    def compute_clearance_cost(
        self, 
        trajectory: np.ndarray,
        return_details: bool = False,
        max_points: Optional[int] = None,
        discount_factor: float = 0.95
    ) -> Union[float, Tuple[float, Dict]]:
        """
        计算清距/安全余量惩罚代价（改进版：使用 discount 折扣系数的加权平均）/
        Compute the clearance/safety-margin penalty cost
        (improved version: discount-weighted average using a discount factor).

        E_clear = Σ (γ^i * ψ(d_i)) / Σ γ^i, 其中：
        - γ = discount_factor (0.9) - 折扣系数，让前面的点权重更高
        - ψ(d) = ReLU(m-d)² 当 d < m（太近，安全惩罚）
        - ψ(d) = ReLU(d-optimal_max)² 当 d > optimal_max（太远，效率惩罚）
        - ψ(d) = 0 当 m ≤ d ≤ optimal_max（理想范围）

        E_clear = Σ (γ^i * ψ(d_i)) / Σ γ^i, where:
        - γ = discount_factor (0.9): discount factor giving earlier points higher weight.
        - ψ(d) = ReLU(m-d)² when d < m (too close, safety penalty).
        - ψ(d) = ReLU(d-optimal_max)² when d > optimal_max (too far, efficiency penalty).
        - ψ(d) = 0 when m ≤ d ≤ optimal_max (ideal range).

        Args:
            trajectory: 轨迹点 [M, 3]，每行为 (x, y, z) / trajectory points [M, 3], each row is (x, y, z).
            return_details: 是否返回详细信息 / whether to return detailed information.
            max_points: 最大计算点数，None 表示计算所有点，设置为 10 只计算前 10 个点 /
                maximum number of points to compute; None means all points, e.g. 10 computes only the first 10 points.
            discount_factor: 折扣系数（默认 0.9），用于对后续点进行衰减 /
                discount factor (default 0.9) used to decay the weight of later points.

        Returns:
            clearance_cost: 清距/安全余量惩罚代价 / clearance/safety-margin penalty cost.
            details: (可选) 详细信息字典 / (optional) dictionary of detailed information.
        """
        M = len(trajectory)
        if M == 0:
            return 0.0 if not return_details else (0.0, {})
        
        # 根据 max_points 参数限制计算的点数
        # Limit the number of computed points according to the max_points argument
        if max_points is not None and max_points > 0:
            compute_trajectory = trajectory[:max_points]
            M_compute = len(compute_trajectory)
        else:
            compute_trajectory = trajectory
            M_compute = M

        # 查询每个点的 ESDF 距离值 - 使用批量查询（如果可用）
        # Query the ESDF distance for each point, using batch query if available
        if hasattr(self.esdf_query, 'batch'):
            # 使用批量查询，一次性查询所有点 / use batch query to look up all points at once
            distances = self.esdf_query.batch(compute_trajectory)
        else:
            # 回退到逐点查询 / fall back to point-by-point query
            distances = np.array([self.esdf_query(point) for point in compute_trajectory])

        # 定义理想距离范围 / define the ideal distance range
        optimal_min = self.safety_margin  # 安全余量（例如 0.25m）/ safety margin (e.g. 0.25m)
        # 最大理想距离（例如 0.5m），不要过远，保持紧凑路径
        # Maximum ideal distance (e.g. 0.5m); avoid going too far to keep the path compact
        optimal_max = 0.6

        # 计算清距/安全余量惩罚：双向惩罚
        # Compute the clearance/safety-margin penalty: a two-sided penalty
        # 1. 安全惩罚：距离过近 (d < optimal_min) / safety penalty: too close (d < optimal_min)
        safety_violations = np.maximum(0, optimal_min - distances)  # ReLU(m-d)
        safety_penalties = safety_violations ** 2  # 平方惩罚，权重较高 / squared penalty with higher weight

        # 2. 效率惩罚：距离过远 (d > optimal_max) / efficiency penalty: too far (d > optimal_max)
        efficiency_violations = np.maximum(0, distances - optimal_max)  # ReLU(d-optimal_max)
        efficiency_penalties = 0.1 * (efficiency_violations ** 2)  # 较低权重的平方惩罚 / squared penalty with lower weight
        # 总惩罚 = 安全惩罚 + 效率惩罚 / total penalty = safety penalty + efficiency penalty
        total_penalties = safety_penalties + efficiency_penalties

        # 使用 discount 折扣系数计算加权平均（类似强化学习）
        # Compute a discount-weighted average (analogous to reinforcement learning)
        # 权重: [1, γ, γ², γ³, ...] 前面的点权重更高 / weights [1, γ, γ², γ³, ...] give earlier points higher weight
        discount_weights = np.power(discount_factor, np.arange(M_compute))

        # 加权惩罚求和除以权重求和（归一化的加权平均）
        # Sum of weighted penalties divided by the sum of weights (normalized weighted average)
        weighted_penalties_sum = np.sum(total_penalties * discount_weights)
        weights_sum = np.sum(discount_weights)
        clearance_cost = weighted_penalties_sum / weights_sum if weights_sum > 0 else 0.0

        if return_details:
            # 计算在理想范围内的点数 / count the points within the ideal range
            in_optimal_range = np.logical_and(distances >= optimal_min, distances <= optimal_max)
            
            details = {
                'distances': distances,
                'safety_violations': safety_violations,
                'efficiency_violations': efficiency_violations,
                'safety_penalties': safety_penalties,
                'efficiency_penalties': efficiency_penalties,
                'total_penalties': total_penalties,
                'discount_weights': discount_weights,
                'discount_factor': discount_factor,
                'weighted_penalties': total_penalties * discount_weights,
                'num_safety_violations': np.sum(safety_violations > 0),
                'num_efficiency_violations': np.sum(efficiency_violations > 0),
                'num_optimal_range': np.sum(in_optimal_range),
                'max_safety_violation': np.max(safety_violations),
                'max_efficiency_violation': np.max(efficiency_violations),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'safety_violation_ratio': np.sum(safety_violations > 0) / M_compute,
                'efficiency_violation_ratio': np.sum(efficiency_violations > 0) / M_compute,
                'optimal_range_ratio': np.sum(in_optimal_range) / M_compute,
                'optimal_min': optimal_min,
                'optimal_max': optimal_max,
                'total_trajectory_points': M,
                'computed_points': M_compute,
                'max_points_limit': max_points
            }
            return clearance_cost, details
        
        return clearance_cost

        if return_details:
            # 计算在理想范围内的点数 / count the points within the ideal range
            in_optimal_range = np.logical_and(distances >= optimal_min, distances <= optimal_max)

            details = {
                'distances': distances,
                'safety_violations': safety_violations,
                'efficiency_violations': efficiency_violations,
                'safety_penalties': safety_penalties,
                'efficiency_penalties': efficiency_penalties,
                'total_penalties': total_penalties,
                'num_safety_violations': np.sum(safety_violations > 0),
                'num_efficiency_violations': np.sum(efficiency_violations > 0),
                'num_optimal_range': np.sum(in_optimal_range),
                'max_safety_violation': np.max(safety_violations),
                'max_efficiency_violation': np.max(efficiency_violations),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'safety_violation_ratio': np.sum(safety_violations > 0) / M_compute,
                'efficiency_violation_ratio': np.sum(efficiency_violations > 0) / M_compute,
                'optimal_range_ratio': np.sum(in_optimal_range) / M_compute,
                'optimal_min': optimal_min,
                'optimal_max': optimal_max,
                'total_trajectory_points': M,
                'computed_points': M_compute,
                'max_points_limit': max_points
            }
            return clearance_cost, details
        
        return clearance_cost
    
    def compute_length_cost(
        self, 
        trajectory: np.ndarray,
        return_details: bool = False
    ) -> Union[float, Tuple[float, Dict]]:
        """
        计算轨迹长度代价 / Compute the trajectory length cost.

        E_len = (1/(M-1)) * Σ ||p_{i+1} - p_i||

        Args:
            trajectory: 轨迹点 [M, 3] / trajectory points [M, 3].
            return_details: 是否返回详细信息 / whether to return detailed information.

        Returns:
            length_cost: 轨迹长度代价 / trajectory length cost.
            details: (可选) 详细信息 / (optional) detailed information.
        """
        M = len(trajectory)
        if M <= 1:
            return 0.0 if not return_details else (0.0, {})

        # 计算相邻点之间的距离 / compute distances between adjacent points
        segment_lengths = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)

        # 平均段长度 / mean segment length
        length_cost = np.mean(segment_lengths)
        
        if return_details:
            details = {
                'segment_lengths': segment_lengths,
                'total_length': np.sum(segment_lengths),
                'num_segments': len(segment_lengths),
                'max_segment': np.max(segment_lengths),
                'min_segment': np.min(segment_lengths),
                'std_segment': np.std(segment_lengths)
            }
            return length_cost, details
        
        return length_cost
    
    def compute_goal_cost(
        self, 
        trajectory: np.ndarray, 
        goal_point: np.ndarray,
        return_details: bool = False
    ) -> Union[float, Tuple[float, Dict]]:
        """
        计算终点误差代价 / Compute the goal endpoint error cost.

        E_goal = ||p_M - g||²

        Args:
            trajectory: 轨迹点 [M, 3] / trajectory points [M, 3].
            goal_point: 目标点 [3] / goal point [3].
            return_details: 是否返回详细信息 / whether to return detailed information.

        Returns:
            goal_cost: 终点误差代价 / goal endpoint error cost.
            details: (可选) 详细信息 / (optional) detailed information.
        """
        if len(trajectory) == 0:
            return float('inf') if not return_details else (float('inf'), {})

        # 获取轨迹终点 / get the trajectory endpoint
        end_point = trajectory[-1]

        # 计算欧氏距离的平方 / compute the squared Euclidean distance
        goal_error = np.linalg.norm(end_point - goal_point)
        goal_cost = goal_error ** 2
        
        if return_details:
            details = {
                'end_point': end_point,
                'goal_point': goal_point,
                'goal_error': goal_error,
                'goal_error_squared': goal_cost,
                'error_components': end_point - goal_point
            }
            return goal_cost, details
        
        return goal_cost
    
    def evaluate_trajectory(
        self,
        trajectory: np.ndarray,
        goal_point: np.ndarray,
        return_components: bool = False,
        clearance_max_points: Optional[int] = None
    ) -> Union[float, Dict]:
        """
        评估单条轨迹的总代价 / Evaluate the total cost of a single trajectory.

        Args:
            trajectory: 轨迹点 [M, 3] / trajectory points [M, 3].
            goal_point: 目标点 [3] / goal point [3].
            return_components: 是否返回各项代价的详细分解 / whether to return the detailed breakdown of each cost term.
            clearance_max_points: 清距计算的最大点数，None 表示计算所有点，10 表示只计算前 10 个点 /
                maximum number of points for the clearance computation; None means all points, 10 means only the first 10.

        Returns:
            total_cost: 总代价 (return_components=False) / total cost (return_components=False).
            cost_dict: 代价详细分解 (return_components=True) / detailed cost breakdown (return_components=True).
        """
        # 计算各项代价 / compute each cost term
        clearance_cost, clearance_details = self.compute_clearance_cost(
            trajectory, return_details=True, max_points=clearance_max_points
        )
        length_cost, length_details = self.compute_length_cost(trajectory, return_details=True)
        goal_cost, goal_details = self.compute_goal_cost(trajectory, goal_point, return_details=True)

        # 加权求和 / weighted sum
        total_cost = (
            self.weights['clear'] * clearance_cost +
            self.weights['length'] * length_cost +
            self.weights['goal'] * goal_cost
        )
        
        if return_components:
            return {
                'total_cost': total_cost,
                'clearance_cost': clearance_cost,
                'length_cost': length_cost,
                'goal_cost': goal_cost,
                'weights': self.weights.copy(),
                'weighted_costs': {
                    'clearance': self.weights['clear'] * clearance_cost,
                    'length': self.weights['length'] * length_cost,
                    'goal': self.weights['goal'] * goal_cost
                },
                'details': {
                    'clearance': clearance_details,
                    'length': length_details,
                    'goal': goal_details
                }
            }
        
        return total_cost
    
    def evaluate_trajectories(
        self,
        trajectories: List[np.ndarray],
        goal_points: Union[np.ndarray, List[np.ndarray]],
        return_ranking: bool = True,
        clearance_max_points: Optional[int] = None
    ) -> Dict:
        """
        评估多条候选轨迹 / Evaluate multiple candidate trajectories.

        Args:
            trajectories: 候选轨迹列表 / list of candidate trajectories.
            goal_points: 目标点，可以是单个点或每条轨迹对应一个目标点 /
                goal point(s); either a single point or one goal point per trajectory.
            return_ranking: 是否返回轨迹排名 / whether to return the trajectory ranking.
            clearance_max_points: 清距计算的最大点数，None 表示计算所有点，10 表示只计算前 10 个点 /
                maximum number of points for the clearance computation; None means all points, 10 means only the first 10.

        Returns:
            evaluation_results: 评估结果字典 / dictionary of evaluation results.
        """
        num_trajectories = len(trajectories)

        # 处理目标点 / handle the goal point(s)
        if isinstance(goal_points, np.ndarray) and goal_points.ndim == 1:
            # 单个目标点，复制给所有轨迹 / a single goal point, replicated for all trajectories
            goal_points_list = [goal_points] * num_trajectories
        else:
            goal_points_list = goal_points

        # 评估每条轨迹 / evaluate each trajectory
        results = []
        for i, (trajectory, goal) in enumerate(zip(trajectories, goal_points_list)):
            cost_info = self.evaluate_trajectory(
                trajectory, goal, return_components=True, clearance_max_points=clearance_max_points
            )
            cost_info['trajectory_id'] = i
            results.append(cost_info)
        
        # 排序（分数越小越好）/ sort (lower cost is better)
        if return_ranking:
            results.sort(key=lambda x: x['total_cost'])

        # 汇总统计 / summary statistics
        total_costs = [r['total_cost'] for r in results]
        clearance_costs = [r['clearance_cost'] for r in results]
        length_costs = [r['length_cost'] for r in results]
        goal_costs = [r['goal_cost'] for r in results]
        
        summary = {
            'num_trajectories': num_trajectories,
            'best_cost': min(total_costs) if total_costs else float('inf'),
            'worst_cost': max(total_costs) if total_costs else float('inf'),
            'mean_cost': np.mean(total_costs) if total_costs else float('inf'),
            'std_cost': np.std(total_costs) if total_costs else 0.0,
            'cost_components_stats': {
                'clearance': {'mean': np.mean(clearance_costs), 'std': np.std(clearance_costs)},
                'length': {'mean': np.mean(length_costs), 'std': np.std(length_costs)},
                'goal': {'mean': np.mean(goal_costs), 'std': np.std(goal_costs)}
            }
        }
        
        return {
            'results': results,
            'summary': summary,
            'weights': self.weights.copy(),
            'safety_margin': self.safety_margin
        }
    
    def visualize_evaluation(
        self,
        evaluation_results: Dict,
        save_path: Optional[str] = None,
        show_top_n: int = 5
    ) -> plt.Figure:
        """
        可视化轨迹评估结果 / Visualize the trajectory evaluation results.

        Args:
            evaluation_results: evaluate_trajectories 的返回结果 / the return value of evaluate_trajectories.
            save_path: 保存路径 / save path.
            show_top_n: 显示前 N 个最佳轨迹的详细信息 / show detailed information for the top-N best trajectories.

        Returns:
            matplotlib 图表对象 / the matplotlib figure object.
        """
        results = evaluation_results['results']
        summary = evaluation_results['summary']
        weights = evaluation_results['weights']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. 总代价分布 / total cost distribution
        total_costs = [r['total_cost'] for r in results]
        ax = axes[0, 0]
        ax.hist(total_costs, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(summary['mean_cost'], color='red', linestyle='--', label=f'Mean: {summary["mean_cost"]:.3f}')
        ax.axvline(summary['best_cost'], color='green', linestyle='--', label=f'Best: {summary["best_cost"]:.3f}')
        ax.set_xlabel('Total Cost')
        ax.set_ylabel('Frequency')
        ax.set_title('Total Cost Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 代价成分对比 / cost component comparison
        ax = axes[0, 1]
        clearance_costs = [r['weighted_costs']['clearance'] for r in results]
        length_costs = [r['weighted_costs']['length'] for r in results]
        goal_costs = [r['weighted_costs']['goal'] for r in results]

        trajectory_ids = range(len(results))
        width = 0.25
        x = np.arange(min(len(results), 20))  # 只显示前 20 个 / show only the first 20
        
        if len(results) <= 20:
            ax.bar(x - width, clearance_costs[:20], width, label='Clearance', alpha=0.8)
            ax.bar(x, length_costs[:20], width, label='Length', alpha=0.8)
            ax.bar(x + width, goal_costs[:20], width, label='Goal', alpha=0.8)
            ax.set_xlabel('Trajectory ID')
            ax.set_xticks(x)
            ax.set_xticklabels([str(results[i]['trajectory_id']) for i in range(min(len(results), 20))])
        else:
            # 太多轨迹时显示统计 / show statistics when there are too many trajectories
            means = [np.mean(clearance_costs), np.mean(length_costs), np.mean(goal_costs)]
            stds = [np.std(clearance_costs), np.std(length_costs), np.std(goal_costs)]
            categories = ['Clearance', 'Length', 'Goal']
            
            ax.bar(categories, means, yerr=stds, alpha=0.8, capsize=5)
            ax.set_xlabel('Cost Component')
        
        ax.set_ylabel('Weighted Cost')
        ax.set_title('Cost Components Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 清距/安全余量违规分析 / clearance/safety-margin violation analysis
        ax = axes[0, 2]
        violation_ratios = [r['details']['clearance']['violation_ratio'] for r in results]
        max_violations = [r['details']['clearance']['max_violation'] for r in results]
        
        scatter = ax.scatter(violation_ratios, max_violations, 
                           c=total_costs, cmap='viridis', alpha=0.7)
        ax.set_xlabel('Violation Ratio')
        ax.set_ylabel('Max Violation (m)')
        ax.set_title('Clearance Violation Analysis')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Total Cost')
        
        # 4. 轨迹长度 vs 终点误差 / trajectory length vs goal endpoint error
        ax = axes[1, 0]
        total_lengths = [r['details']['length']['total_length'] for r in results]
        goal_errors = [np.sqrt(r['goal_cost']) for r in results]  # 开根号恢复到距离 / take square root to recover the distance
        
        scatter = ax.scatter(total_lengths, goal_errors, 
                           c=total_costs, cmap='viridis', alpha=0.7)
        ax.set_xlabel('Trajectory Length (m)')
        ax.set_ylabel('Goal Error (m)')
        ax.set_title('Length vs Goal Error')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Total Cost')
        
        # 5. 最佳轨迹详细信息 / detailed information of the best trajectories
        ax = axes[1, 1]
        if results:
            best_results = results[:show_top_n]
            
            text_info = f"Evaluation Summary:\n"
            text_info += f"Total Trajectories: {summary['num_trajectories']}\n"
            text_info += f"Safety Margin: {evaluation_results['safety_margin']:.2f}m\n\n"
            text_info += f"Weights: Clear={weights['clear']}, Len={weights['length']}, Goal={weights['goal']}\n\n"
            text_info += f"Top {len(best_results)} Trajectories:\n"
            
            for i, result in enumerate(best_results):
                tid = result['trajectory_id']
                total = result['total_cost']
                clear = result['clearance_cost']
                length = result['length_cost']
                goal = result['goal_cost']
                violations = result['details']['clearance']['num_violations']
                
                text_info += f"{i+1}. ID{tid}: Cost={total:.3f}\n"
                text_info += f"   Clear={clear:.3f}, Len={length:.2f}, Goal={goal:.3f}\n"
                text_info += f"   Violations: {violations}\n"
        else:
            text_info = "No trajectories to evaluate."
        
        ax.text(0.05, 0.95, text_info, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9, fontfamily='monospace')
        ax.axis('off')
        ax.set_title('Evaluation Summary')
        
        # 6. 权重敏感性分析 / weight sensitivity analysis
        ax = axes[1, 2]
        if len(results) > 1:
            # 显示不同权重对排名的影响 / show how different weights affect the ranking
            original_ranking = [r['trajectory_id'] for r in results[:10]]

            # 测试只考虑清距/安全余量的排名 / test the ranking considering only the clearance/safety-margin cost
            clear_only_results = sorted(results, key=lambda x: x['clearance_cost'])
            clear_ranking = [r['trajectory_id'] for r in clear_only_results[:10]]

            # 计算排名相关性 / compute the ranking correlation
            rank_changes = []
            for i, tid in enumerate(original_ranking):
                if tid in clear_ranking:
                    new_rank = clear_ranking.index(tid)
                    rank_changes.append(abs(i - new_rank))
                else:
                    rank_changes.append(10)  # 排名变化很大 / large change in ranking
            
            ax.bar(range(len(rank_changes)), rank_changes, alpha=0.7)
            ax.set_xlabel('Original Ranking')
            ax.set_ylabel('Rank Change (Clear-only)')
            ax.set_title('Ranking Sensitivity to Weights')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Need >1 trajectory\nfor sensitivity analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Ranking Sensitivity Analysis')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"评估结果可视化保存到: {save_path}")
        
        return fig


def create_mock_esdf_query(depth_image: np.ndarray, camera_intrinsics: Dict) -> Callable:
    """
    创建一个基于深度图像的模拟 ESDF 查询函数 / Create a mock ESDF query function based on a depth image.

    Args:
        depth_image: 深度图像 [H, W]，值为深度（米）/ depth image [H, W] with values in meters.
        camera_intrinsics: 相机内参字典 / dictionary of camera intrinsics.

    Returns:
        esdf_query_function: ESDF 查询函数 / the ESDF query function.
    """
    def esdf_query(point_3d: np.ndarray) -> float:
        """
        查询 3D 点的 ESDF 值（距离最近障碍物的距离）/
        Query the ESDF value of a 3D point (distance to the nearest obstacle).

        Args:
            point_3d: 3D 点坐标 [x, y, z] / 3D point coordinates [x, y, z].

        Returns:
            distance: 到最近障碍物的距离（正值为自由空间，负值为障碍物内部）/
                distance to the nearest obstacle (positive in free space, negative inside an obstacle).
        """
        x, y, z = point_3d

        # 将 3D 点投影到图像平面 / project the 3D point onto the image plane
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        ppx = camera_intrinsics['ppx']
        ppy = camera_intrinsics['ppy']
        
        if z <= 0.01:  # 避免除零 / avoid division by zero
            return 0.1  # 很近的障碍物 / a very close obstacle

        u = int((x * fx) / z + ppx)
        v = int((y * fy) / z + ppy)

        H, W = depth_image.shape

        # 检查投影是否在图像范围内 / check whether the projection lies within the image
        if u < 0 or u >= W or v < 0 or v >= H:
            return 1.0  # 图像外认为是自由空间 / outside the image is treated as free space

        # 获取该像素的深度值 / get the depth value of this pixel
        pixel_depth = depth_image[v, u]

        # 计算点到像素表面的距离 / compute the distance from the point to the pixel surface
        if pixel_depth <= 0.01:  # 无效深度 / invalid depth
            return 1.0  # 认为是自由空间 / treated as free space

        # 简单的距离计算：点的深度与像素深度的差值
        # Simple distance estimate: the difference between the point depth and the pixel depth
        distance_to_surface = z - pixel_depth

        # 考虑最小安全距离 / enforce a minimum safety distance
        min_distance = 0.05  # 5cm 最小距离 / 5 cm minimum distance
        
        return max(distance_to_surface, min_distance)
    
    return esdf_query


def demo_trajectory_evaluation():
    """
    演示轨迹评估功能 / Demonstrate the trajectory evaluation functionality.
    """
    print("=== 轨迹评估演示 ===")

    # 创建模拟数据 / create mock data
    # 1. 创建简单的深度图像（模拟楼梯环境）/ create a simple depth image (simulating a staircase environment)
    H, W = 240, 320
    depth_image = np.ones((H, W)) * 3.0  # 默认 3 米深度 / default depth of 3 meters

    # 添加一些障碍物（楼梯台阶）/ add some obstacles (staircase steps)
    for i in range(3):
        y_start = 80 + i * 40
        y_end = y_start + 30
        depth_image[y_start:y_end, :] = 2.0 - i * 0.3  # 递进的台阶 / progressively closer steps

    # 2. 相机参数 / camera intrinsics
    camera_intrinsics = {
        'fx': 194.776,
        'fy': 194.776, 
        'ppx': 160.0,
        'ppy': 120.0
    }
    
    # 3. 创建 ESDF 查询函数 / create the ESDF query function
    esdf_query = create_mock_esdf_query(depth_image, camera_intrinsics)

    # 4. 创建评估器 / create the evaluator
    evaluator = TrajectoryEvaluator(
        esdf_query_function=esdf_query,
        safety_margin=0.3,
        weights={'clear': 10.0, 'length': 1.0, 'goal': 5.0}
    )
    
    # 5. 创建几条测试轨迹 / create a few test trajectories
    goal_point = np.array([2.0, 0.0, 0.0])  # 目标点：前进 2 米 / goal point: 2 meters ahead

    # 轨迹 1：直线（可能撞到障碍物）/ trajectory 1: straight line (may hit an obstacle)
    traj1 = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0], 
        [1.5, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])
    
    # 轨迹 2：左绕行 / trajectory 2: detour to the left
    traj2 = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.3, 0.0],
        [1.0, 0.5, 0.0],
        [1.5, 0.3, 0.0],
        [2.0, 0.0, 0.0]
    ])
    
    # 轨迹 3：右绕行 / trajectory 3: detour to the right
    traj3 = np.array([
        [0.0, 0.0, 0.0],
        [0.5, -0.3, 0.0],
        [1.0, -0.5, 0.0],
        [1.5, -0.3, 0.0],
        [2.0, 0.0, 0.0]
    ])
    
    # 轨迹 4：过长的绕行 / trajectory 4: an overly long detour
    traj4 = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.8, 0.0],
        [0.8, 1.2, 0.0],
        [1.3, 0.8, 0.0],
        [1.8, 0.3, 0.0],
        [2.0, 0.0, 0.0]
    ])
    
    trajectories = [traj1, traj2, traj3, traj4]
    
    # 6. 评估轨迹 / evaluate the trajectories
    results = evaluator.evaluate_trajectories(trajectories, goal_point)

    # 7. 输出结果 / print the results
    print(f"\n评估结果:")
    print(f"轨迹数量: {results['summary']['num_trajectories']}")
    print(f"最佳代价: {results['summary']['best_cost']:.3f}")
    print(f"平均代价: {results['summary']['mean_cost']:.3f}")
    
    print(f"\n轨迹排名 (代价越小越好):")
    for i, result in enumerate(results['results']):
        tid = result['trajectory_id']
        total = result['total_cost']
        clear = result['clearance_cost']
        length = result['length_cost']  
        goal = result['goal_cost']
        violations = result['details']['clearance']['num_violations']
        
        print(f"{i+1}. 轨迹{tid+1}: 总代价={total:.3f}")
        print(f"   清距={clear:.3f}, 长度={length:.3f}, 终点={goal:.3f}")
        print(f"   违规点数: {violations}")
    
    # 8. 可视化 / visualization
    fig = evaluator.visualize_evaluation(results, show_top_n=4)
    plt.show()
    
    return results, evaluator


if __name__ == "__main__":
    demo_trajectory_evaluation()
