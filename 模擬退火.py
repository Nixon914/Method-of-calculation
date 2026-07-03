import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import cec2017.functions as functions
def simulated_annealing(objective_func, dim, max_iter=100, T_init=1000, T_min=1e-5, alpha=0.99):
    # 初始化参数
    current_solution = np.random.uniform(-100, 100, dim)
    current_cost = objective_func(np.array([current_solution]))[0]
    
    # 初始化最佳解和最佳目标函数值
    best_solution = current_solution.copy()
    best_cost = current_cost
    
    # 模拟退火过程
    for t in range(max_iter):
        T = T_init * (alpha ** t)
        if T < T_min:
            break

        # 随机扰动当前解生成新解
        new_solution = current_solution + np.random.uniform(-1, 1, dim)
        new_cost = objective_func(np.array([new_solution]))[0]

        # Metropolis准则决定是否接受新解
        if np.random.rand() < np.exp((current_cost - new_cost) / T):
            current_solution = new_solution.copy()
            current_cost = new_cost

        # 更新最佳解和最佳目标函数值
        if new_cost < best_cost:
            best_solution = new_solution.copy()
            best_cost = new_cost
    
    return best_solution, best_cost


def test_simulated_annealing_on_cec2017():
    # 共用參數
    dim = 10 
    max_iter = 1000
    runs = 30
    results = []
    
    for i, f in enumerate(functions.all_functions):
        function_results = []
        
        for run in range(runs):
            best_solution, best_fitness = simulated_annealing(f, dim, max_iter=max_iter)
            function_results.append(best_fitness)
            print(f"SA Function f{i+1}, Run {run + 1}: Best Fitness: {best_fitness}")
        
        results.append(function_results)
        
    return results

results_sa = test_simulated_annealing_on_cec2017()
