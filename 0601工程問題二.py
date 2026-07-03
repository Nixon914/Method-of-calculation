import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import rcParams

# 設定 matplotlib 字體，避免中文字無法顯示
rcParams['font.sans-serif'] = ['Taipei Sans TC Beta', 'Microsoft JhengHei', 'SimHei']
rcParams['axes.unicode_minus'] = False  # 確保負號正常顯示

# 定義彈簧設計問題的目標函數
def spring_design_cost(x):
    """
    彈簧設計問題的目標函數和約束條件
    x1: 線徑(d)
    x2: 彈簧平均直徑(D)
    x3: 活動圈數(N)
    """
    x = x.flatten()
    x1, x2, x3 = x[0], x[1], x[2]
    
    # 目標函數
    cost = (x3 + 2) * x2 * x1 * x1
    
    # 處理約束
    penalty = 0
    # 約束1
    g1 = 1 - x3 * x2**3 / (71785 * x1**4)
    if g1 > 0:
        penalty += 1e6 * g1
    # 約束2
    g2 = (4 * x2**2 - x1 * x2) / (12566 * (x2 * x1**3 - x1**4)) + 1 / (5108 * x1**2) - 1
    if g2 > 0:
        penalty += 1e6 * g2
    # 約束3
    g3 = 1 - 140.45 * x1 / (x2**2 * x3)
    if g3 > 0:
        penalty += 1e6 * g3
    # 約束4
    g4 = (x1 + x2) / 1.5 - 1
    if g4 > 0:
        penalty += 1e6 * g4
    
    return cost + penalty

# 模拟退火局部优化函数 (彈簧設計版本)
def simulated_annealing(f, current_pos, current_score, lb, ub, T0, T_min, cooling_rate):
    dim = current_pos.shape[0]
    T = T0
    best_pos = current_pos.copy()
    best_score = current_score

    while T > T_min:
        # 生成一个新的邻域解，使用相對擾動避免超出邊界
        perturbation = np.random.uniform(-0.1, 0.1, dim)
        # 對所有變數使用相對擾動 (彈簧設計都是連續變數)
        perturbation = perturbation * (ub - lb) * 0.05  # 使用5%的範圍擾動
        
        new_pos = best_pos + perturbation
        new_pos = np.clip(new_pos, lb, ub)
        
        # 计算新解的适应度
        new_score = f(new_pos.reshape(1, -1))
        
        # 如果新解更优，直接接受
        if new_score < best_score:
            best_pos, best_score = new_pos, new_score
        else:
            # 以一定概率接受较差解
            acceptance_probability = np.exp(-(new_score - best_score) / T)
            if np.random.rand() < acceptance_probability:
                best_pos, best_score = new_pos, new_score
        
        # 降低温度
        T *= cooling_rate

    return best_pos, best_score

def levy_flight(dim, sigma=0.01):
    beta = 1.5
    sigma_u = np.power(
        np.random.gamma(1 + beta, 1) * np.sin(np.pi * beta / 2) / 
        (np.random.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)),
        1 / beta
    )
    u = np.random.normal(0, sigma_u, dim)
    v = np.random.normal(0, 1, dim)
    step = sigma * u / np.power(np.abs(v), 1/beta)
    return step

# 二次插值
def quadratic_interpolation(alpha_pos, random1_pos, random2_pos, alpha_score, random1_score, random2_score, lb=None, ub=None):
    # 確保輸入是二維數組
    if alpha_pos.ndim == 1:
        alpha_pos = alpha_pos.reshape(1, -1)
    if random1_pos.ndim == 1:
        random1_pos = random1_pos.reshape(1, -1)
    if random2_pos.ndim == 1:
        random2_pos = random2_pos.reshape(1, -1)
    
    # 獲取維度
    n_dim = alpha_pos.shape[1]
    
    # 初始化K_q，確保是二維數組
    K_q = np.zeros((1, n_dim))
    
    # 對每個維度q計算K_q
    for q in range(n_dim):
        X_q = alpha_pos[0, q]
        Y_q = random1_pos[0, q]
        Z_q = random2_pos[0, q]
        
        # 計算分子
        numerator = (Z_q**2 - Y_q**2) * alpha_score + (X_q**2 - Z_q**2) * random1_score + (Y_q**2 - X_q**2) * random2_score
        
        # 計算分母
        denominator = 2 * ((Z_q - Y_q) * alpha_score + (X_q - Z_q) * random1_score + (Y_q - X_q) * random2_score)
        
        # 避免除零
        if abs(denominator) < 1e-10:
            K_q[0, q] = X_q  # 如果分母接近零，保持alpha位置
        else:
            interpolated_value = float(numerator / denominator)
            # 確保插值結果在合理範圍內
            K_q[0, q] = max(min(interpolated_value, ub[q] if q < len(ub) else 1000), 
                           lb[q] if q < len(lb) else -1000)
    
    return K_q

# GWO-AFSA 主算法，结合模拟退火，適用於彈簧設計問題
def gwo_afsa_with_worst_update(f, dim, lb, ub, n_wolves, max_iter):
    # 初始化种群
    X = np.random.uniform(lb, ub, (n_wolves, dim))
    
    # 確保所有個體都在邊界內 (彈簧設計都是連續變數，無需離散處理)
    X = np.clip(X, lb, ub)
    
    alpha_pos, beta_pos, delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    fitness_counter, kqc, sac, fitness_counter_OUT, worstc = 0, 0, 0, 0, 0
    
    # 模拟退火参数
    T0, T_min, cooling_rate = 100, 1e-3, 0.9
    
    # 記錄收斂歷史
    best_scores_history = []

    for t in range(max_iter):
        # 先計算一次適應值
        for i in range(n_wolves):
            fitness = f(X[i, :].reshape(1, -1))
            if np.isinf(fitness):
                fitness = 1e10
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, X[i, :].copy()
                fitness_counter_OUT += 1
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, X[i, :].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, X[i, :].copy()
        
        # 计算收敛因子 a
        a = 2 - t * (2 / max_iter)

        for i in range(n_wolves):
            fitness = f(X[i, :].reshape(1, -1))
            if np.isinf(fitness):
                fitness = 1e10
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, X[i, :].copy()
                fitness_counter += 1
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, X[i, :].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, X[i, :].copy()
            
            fitness = np.array([f(x.reshape(1, -1)) for x in X])
            worst_indices = np.argsort(fitness)[-3:]  # 找到适应度最差的3个个体

            for idx in worst_indices:
                # 使用相對較小的Lévy飛行步長，避免超出邊界
                levy_step = levy_flight(dim, sigma=0.01)  # 彈簧設計使用稍大的sigma值
                new_pos = X[idx] + levy_step
                
                new_pos = np.clip(new_pos, lb, ub)  # 確保在邊界內
                new_fitness = f(new_pos.reshape(1, -1))
                
                if new_fitness < fitness[idx]:
                    X[idx] = new_pos
                    fitness[idx] = new_fitness
                    worstc += 1

            # 步骤 9更新A、C 
            r1, r2, r3 = np.random.rand(dim), np.random.rand(dim), np.random.rand(dim)
            A1, A2, A3 = 2 * a * r1 - a, 2 * a * r2 - a, 2 * a * r3 - a
            C1, C2, C3 = 2 * np.random.rand(dim), 2 * np.random.rand(dim), 2 * np.random.rand(dim)

            D_alpha = abs(C1 * alpha_pos - X[i, :])
            D_beta = abs(C2 * beta_pos - X[i, :])
            D_delta = abs(C3 * delta_pos - X[i, :])

            # 步骤 10根據狼群中每只狼的距離選擇最接近的狼進行位置更新
            if np.min(D_alpha) <= np.min(D_beta) and np.min(D_alpha) <= np.min(D_delta):
                X[i, :] = alpha_pos - A1 * D_alpha
            elif np.min(D_beta) <= np.min(D_alpha) and np.min(D_beta) <= np.min(D_delta):
                X[i, :] = beta_pos - A2 * D_beta
            else:
                X[i, :] = delta_pos - A3 * D_delta

            # 處理x1和x2的離散約束
            # 彈簧設計都是連續變數，無需離散處理
            
            # 將更新後的位置限制在上下界之內
            X[i, :] = np.clip(X[i, :], lb, ub)

            fitness = np.array([f(X[l, :].reshape(1, -1)) for l in range(n_wolves)])
        
        # 步驟14 二次插值
        random_indices = np.random.choice(X.shape[0], 2, replace=False)
        random1_pos = X[random_indices[0], :]
        random2_pos = X[random_indices[1], :]
        random1_score = f(random1_pos.reshape(1, -1))
        random2_score = f(random2_pos.reshape(1, -1))

        new_wolf = quadratic_interpolation(alpha_pos, random1_pos, random2_pos, alpha_score, random1_score, random2_score, lb, ub)
        
        # 彈簧設計都是連續變數，無需離散處理，只需確保邊界約束
        new_wolf = np.clip(new_wolf, lb.reshape(1, -1), ub.reshape(1, -1))  # 確保在邊界內
        
        new_wolf_fitness = f(new_wolf.reshape(1, -1))
        
        # 確保 fitness 的維度正確並更新
        fitness = np.array([f(X[j, :].reshape(1, -1)) for j in range(X.shape[0])])

        # 找到最差狼
        worst_index = np.argmax(fitness)
        worst_fitness = fitness[worst_index]

        # 如果新狼比最差狼表現更好，進行替換
        if new_wolf_fitness < worst_fitness:
            # 確保替換的解在邊界內
            new_wolf_clipped = np.clip(new_wolf.flatten(), lb, ub)
            X[worst_index, :] = new_wolf_clipped
            fitness[worst_index] = new_wolf_fitness
            kqc += 1

        # 每10次迭代執行模擬退火
        if t % 10 == 0:
            Nalpha_score = alpha_score
            temp_alpha_pos, temp_alpha_score = simulated_annealing(f, alpha_pos, alpha_score, lb, ub, T0, T_min, cooling_rate)
            # 確保模擬退火的結果在邊界內
            temp_alpha_pos = np.clip(temp_alpha_pos, lb, ub)
            if temp_alpha_score < Nalpha_score:
                alpha_pos = temp_alpha_pos
                alpha_score = temp_alpha_score
                sac += 1
        
        # 記錄收斂歷史
        best_scores_history.append(alpha_score)
        
        # 輸出進度信息
        if (t+1) % 100 == 0 or t == 0:
            print(f"迭代 {t+1}/{max_iter}, 目前最佳適應度: {alpha_score:.6f}")
            print(f"最佳解: d={alpha_pos[0]:.6f}, D={alpha_pos[1]:.6f}, N={alpha_pos[2]:.6f}")

    # 打印最終統計信息
    print("\n===== 優化完成 =====")
    print(f"最終最佳適應度(重量): {alpha_score:.6f}")
    print(f"最佳設計參數:")
    print(f"  d (線徑) = {alpha_pos[0]:.6f}")
    print(f"  D (彈簧平均直徑) = {alpha_pos[1]:.6f}")
    print(f"  N (活動圈數) = {alpha_pos[2]:.6f}")
    
    # 驗證約束條件
    x = alpha_pos
    x1, x2, x3 = x[0], x[1], x[2]
    g1 = 1 - x3 * x2**3 / (71785 * x1**4)
    g2 = (4 * x2**2 - x1 * x2) / (12566 * (x2 * x1**3 - x1**4)) + 1 / (5108 * x1**2) - 1
    g3 = 1 - 140.45 * x1 / (x2**2 * x3)
    g4 = (x1 + x2) / 1.5 - 1
    
    print("\n約束條件檢查:")
    print(f"  g1 = {g1:.6f} {'✓' if g1 <= 0 else '✗'}")
    print(f"  g2 = {g2:.6f} {'✓' if g2 <= 0 else '✗'}")
    print(f"  g3 = {g3:.6f} {'✓' if g3 <= 0 else '✗'}")
    print(f"  g4 = {g4:.6f} {'✓' if g4 <= 0 else '✗'}")
    
    print("\n算法統計:")
    print(f"  二次插值改進次數: {kqc}")
    print(f"  最差解更新次數: {worstc}")
    print(f"  模擬退火改進次數: {sac}")
    print(f"  Fitness計數器: {fitness_counter}")
    print(f"  Fitness計數器OUT: {fitness_counter_OUT}")

    return alpha_pos, alpha_score, best_scores_history

# 主函數
def main():
    # 設置問題參數
    dim = 3  # 設計變數的數量 (d, D, N)
    # 設計變數的下界和上界
    lb = np.array([0.05, 0.25, 2.0])   # [d_min, D_min, N_min]
    ub = np.array([2.0, 1.3, 15.0])    # [d_max, D_max, N_max]
    
    # 算法參數
    n_wolves = 50  # 狼群數量
    max_iter = 1000  # 最大迭代次數
    
    print("====== 彈簧設計優化 ======")
    print("設計變數: [d, D, N]")
    print(f"變數範圍:")
    print(f"  0.05 ≤ d ≤ 2.0    (線徑)")
    print(f"  0.25 ≤ D ≤ 1.3    (彈簧平均直徑)")
    print(f"  2.0 ≤ N ≤ 15.0    (活動圈數)")
    print(f"\n開始優化...")
    
    # 用於儲存多次運行的結果
    results = []
    best_solutions = []
    execution_times = []
    convergence_histories = []
    
    # 運行30次並收集結果
    for i in range(30):
        print(f"\n========== 運行 {i+1}/30 ==========")
        start_time = time.time()
        best_solution, best_cost, convergence = gwo_afsa_with_worst_update(
            spring_design_cost, dim, lb, ub, n_wolves, max_iter
        )
        end_time = time.time()
        execution_time = end_time - start_time
        
        results.append(best_cost)
        best_solutions.append(best_solution)
        execution_times.append(execution_time)
        convergence_histories.append(convergence)
        print(f"運行 {i+1}/30 完成: 最佳重量 = {best_cost:.6f}, 執行時間 = {execution_time:.2f} 秒")
    
    # 計算統計結果
    best_cost_overall = min(results)
    best_index = results.index(best_cost_overall)
    worst_cost_overall = max(results)
    mean_cost = np.mean(results)
    median_cost = np.median(results)
    std_dev = np.std(results)
    mean_execution_time = np.mean(execution_times)
    
    print("\n===== 總統計結果 =====")
    print(f"最佳重量: {best_cost_overall:.6f}")
    print(f"最差重量: {worst_cost_overall:.6f}")
    print(f"平均重量: {mean_cost:.6f}")
    print(f"中位數重量: {median_cost:.6f}")
    print(f"標準差: {std_dev:.6f}")
    print(f"平均執行時間: {mean_execution_time:.2f} 秒")
    
    # 顯示最佳解的詳細信息
    best_sol = best_solutions[best_index]
    print("\n最佳解的詳細信息:")
    print(f"  d (線徑) = {best_sol[0]:.6f}")
    print(f"  D (彈簧平均直徑) = {best_sol[1]:.6f}")
    print(f"  N (活動圈數) = {best_sol[2]:.6f}")
    
    # 驗證最佳解的約束條件
    x = best_sol
    x1, x2, x3 = x[0], x[1], x[2]
    g1 = 1 - x3 * x2**3 / (71785 * x1**4)
    g2 = (4 * x2**2 - x1 * x2) / (12566 * (x2 * x1**3 - x1**4)) + 1 / (5108 * x1**2) - 1
    g3 = 1 - 140.45 * x1 / (x2**2 * x3)
    g4 = (x1 + x2) / 1.5 - 1
    
    print("\n最佳解約束條件檢查:")
    print(f"  g1 = {g1:.6f} {'✓' if g1 <= 0 else '✗'}")
    print(f"  g2 = {g2:.6f} {'✓' if g2 <= 0 else '✗'}")
    print(f"  g3 = {g3:.6f} {'✓' if g3 <= 0 else '✗'}")
    print(f"  g4 = {g4:.6f} {'✓' if g4 <= 0 else '✗'}")
    
    # 繪製所有收斂曲線
    plt.figure(figsize=(12, 6))
    plt.title('所有運行的收斂曲線')
    plt.xlabel('迭代次數')
    plt.ylabel('最佳適應度(成本)')
    plt.grid(True)
    
    # 為了避免圖形過於擁擠，只顯示部分運行結果或使用透明度
    for i, history in enumerate(convergence_histories):
        plt.plot(history, alpha=0.3, color='blue')
        
    # 添加平均收斂曲線
    avg_convergence = np.mean(np.array(convergence_histories), axis=0)
    plt.plot(avg_convergence, linewidth=2, color='red', label='平均收斂曲線')
    
    # 添加最佳運行的收斂曲線
    best_convergence = convergence_histories[best_index]
    plt.plot(best_convergence, linewidth=2, color='green', label='最佳運行')
    
    plt.legend()
    plt.show()
    
    # 繪製柱狀統計圖
    plt.figure(figsize=(10, 6))
    plt.bar(range(30), results)
    plt.axhline(y=mean_cost, color='r', linestyle='-', label=f'平均值: {mean_cost:.4f}')
    plt.axhline(y=best_cost_overall, color='g', linestyle='-', label=f'最佳值: {best_cost_overall:.4f}')
    plt.title('30次運行的成本結果')
    plt.xlabel('運行次數')
    plt.ylabel('成本')
    plt.legend()
    plt.grid(True, axis='y')
    plt.show()
    
    return best_solutions[best_index], best_cost_overall, convergence_histories

if __name__ == "__main__":
    best_solution, best_cost, convergence_histories = main()