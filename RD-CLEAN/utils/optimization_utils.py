"""
优化工具函数
"""

import numpy as np
from typing import List, Tuple, Callable
from scipy.optimize import minimize, differential_evolution


def check_convergence(current_value: float, previous_value: float, threshold: float = 1e-6) -> bool:
    """检查优化收敛性"""
    if previous_value == 0:
        return False

    relative_change = abs((current_value - previous_value) / previous_value)
    return relative_change < threshold


def generate_random_initial_guess(bounds: List[Tuple[float, float]], num_trials: int = 10) -> List[np.ndarray]:
    """生成随机初始猜测"""
    initial_guesses = []

    for _ in range(num_trials):
        guess = []
        for low, high in bounds:
            guess.append(np.random.uniform(low, high))
        initial_guesses.append(np.array(guess))

    return initial_guesses


def robust_optimization(
    objective_func: Callable, initial_guess: np.ndarray, bounds: List[Tuple[float, float]], method: str = "L-BFGS-B"
) -> Tuple[np.ndarray, float]:
    """鲁棒优化函数"""
    best_result = None
    best_fval = float("inf")

    # 尝试多个初始猜测
    initial_guesses = generate_random_initial_guess(bounds, 5)
    initial_guesses.insert(0, initial_guess)  # 包含原始猜测

    for guess in initial_guesses:
        try:
            result = minimize(
                fun=objective_func, x0=guess, method=method, bounds=bounds, options={"ftol": 1e-9, "maxiter": 1000}
            )

            if result.fun < best_fval:
                best_result = result
                best_fval = result.fun

        except:
            continue

    if best_result is None:
        return initial_guess, float("inf")

    return best_result.x, best_result.fun
