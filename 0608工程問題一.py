import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import rcParams

# 設定 matplotlib 字體，避免中文字無法顯示
rcParams['font.sans-serif'] = ['Taipei Sans TC Beta', 'Microsoft JhengHei', 'SimHei']
rcParams['axes.unicode_minus'] = False  # 確保負號正常顯示

# 定義壓力容器設計問題的目標函數
def pressure_vessel_cost(x):
    """
    計算壓力容器的成本函數
    x: 設計變數 [Ts, Th, R, L]
    Ts: 殼體厚度
    Th: 頭部厚度
    R: 內徑
    L: 圓柱段長度
    """
    x = x.flatten()
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    
    # 目標函數
    cost = 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3**2 + 3.1661 * x1**2 * x4 + 19.84 * x1**2 * x3
    
    # 處理約束
    penalty = 0
    # 約束1: 殼體厚度約束
    g1 = -x1 + 0.0193 * x3
    if g1 > 0:
        penalty += 1e10 * g1
    # 約束2: 頭部厚度約束
    g2 = -x2 + 0.00954 * x3
    if g2 > 0:
        penalty += 1e10 * g2
    # 約束3: 容積約束
    g3 = -np.pi * x3**2 * x4 - (4/3) * np.pi * x3**3 + 1296000
    if g3 > 0:
        penalty += 1e10 * g3
    # 約束4: 長度約束
    g4 = x4 - 240
    if g4 > 0:
        penalty += 1e10 * g4
    
    return cost + penalty

# 模擬退火局部優化函數
def simulated_annealing(f, current_pos, current_score, lb, ub, T0, T_min, cooling_rate):
    dim = current_pos.shape[0]
    T = T0
    best_pos = current_pos.copy()
    best_score = current_score

    while T > T_min:
        # 生成一個新的鄰域解，使用相對擾動避免超出邊界
        perturbation = np.random.uniform(-0.1, 0.1, dim)
        # 對連續變數使用相對擾動
        perturbation[2:] = perturbation[2:] * (ub[2:] - lb[2:]) * 0.05  # R和L使用5%的範圍擾動
        
        new_pos = best_pos + perturbation
        new_pos = np.clip(new_pos, lb, ub)
        
        # 處理x1和x2的離散約束 (0.0625的倍數)
        new_pos[0] = np.round(new_pos[0] / 0.0625) * 0.0625
        new_pos[1] = np.round(new_pos[1] / 0.0625) * 0.0625
        
        # 計算新解的適應度
        new_score = f(new_pos.reshape(1, -1))
        
        # 如果新解更優，直接接受
        if new_score < best_score:
            best_pos, best_score = new_pos, new_score
        else:
            # 以一定概率接受較差解
            acceptance_probability = np.exp(-(new_score - best_score) / T)
            if np.random.rand() < acceptance_probability:
                best_pos, best_score = new_pos, new_score
        
        # 降低溫度
        T *= cooling_rate

    return best_pos, best_score

# 菁英逆向學習
def elite_reverse_learning(X, lb, ub):
    r = np.random.rand(X.shape[0])
    Y = r * (lb + ub) - X
    Y = np.clip(Y, lb, ub)
    return Y

# Lévy飛行函數
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

# GWO-AFSA 主算法，結合模擬退火，適用於壓力容器問題
def gwo_afsa_with_worst_update(f, dim, lb, ub, n_wolves, max_iter):
    # 初始化種群
    X = np.random.uniform(lb, ub, (n_wolves, dim))
    
    # 處理x1和x2的離散約束
    X[:, 0] = np.round(X[:, 0] / 0.0625) * 0.0625
    X[:, 1] = np.round(X[:, 1] / 0.0625) * 0.0625
    
    # 確保所有個體都在邊界內
    X = np.clip(X, lb, ub)
    
    alpha_pos, beta_pos, delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    fitness_counter, kqc, sac, fitness_counter_OUT, worstc, reverse_fitness_c = 0, 0, 0, 0, 0, 0
    
    # 模擬退火參數
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
        
        # 計算收斂因子 a
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
            
            # 步驟 7: 菁英逆向學習
            reverse_position = elite_reverse_learning(alpha_pos, lb, ub)
            reverse_fitness = f(reverse_position.reshape(1, -1))
            # 如果逆向解更好，則更新alpha狼
            if reverse_fitness < alpha_score:
                alpha_score = float(reverse_fitness)
                alpha_pos = reverse_position.copy()
                reverse_fitness_c += 1
            
            # 步驟8: 使用Lévy飛行更新最差個體
            fitness_all = np.array([f(X[j, :].reshape(1, -1)) for j in range(n_wolves)])
            worst_indices = np.argsort(fitness_all)[-3:]  # 找到適應度最差的3個個體

            for idx in worst_indices:
                # 使用相對較小的Lévy飛行步長，避免超出邊界
                levy_step = levy_flight(dim, sigma=0.005)  # 減小sigma值
                new_pos = X[idx] + levy_step
                
                # 處理x1和x2的離散約束
                new_pos[0] = np.round(new_pos[0] / 0.0625) * 0.0625
                new_pos[1] = np.round(new_pos[1] / 0.0625) * 0.0625
                
                new_pos = np.clip(new_pos, lb, ub)  # 確保在邊界內
                new_fitness = f(new_pos.reshape(1, -1))
                
                if new_fitness < fitness_all[idx]:
                    X[idx] = new_pos
                    fitness_all[idx] = new_fitness
                    worstc += 1

            # 步驟 9: 更新A、C 
            r1, r2, r3 = np.random.rand(dim), np.random.rand(dim), np.random.rand(dim)
            A1, A2, A3 = 2 * a * r1 - a, 2 * a * r2 - a, 2 * a * r3 - a
            C1, C2, C3 = 2 * np.random.rand(dim), 2 * np.random.rand(dim), 2 * np.random.rand(dim)

            D_alpha = abs(C1 * alpha_pos - X[i, :])
            D_beta = abs(C2 * beta_pos - X[i, :])
            D_delta = abs(C3 * delta_pos - X[i, :])

            # 步驟 10: 根據狼群中每隻狼的距離選擇最接近的狼進行位置更新
            if np.min(D_alpha) <= np.min(D_beta) and np.min(D_alpha) <= np.min(D_delta):
                X[i, :] = alpha_pos - A1 * D_alpha
            elif np.min(D_beta) <= np.min(D_alpha) and np.min(D_beta) <= np.min(D_delta):
                X[i, :] = beta_pos - A2 * D_beta
            else:
                X[i, :] = delta_pos - A3 * D_delta

            # 處理x1和x2的離散約束
            X[i, 0] = np.round(X[i, 0] / 0.0625) * 0.0625
            X[i, 1] = np.round(X[i, 1] / 0.0625) * 0.0625
            
            # 將更新後的位置限制在上下界之內
            X[i, :] = np.clip(X[i, :], lb, ub)
        
        # 步驟14: 二次插值
        random_indices = np.random.choice(X.shape[0], 2, replace=False)
        random1_pos = X[random_indices[0], :]
        random2_pos = X[random_indices[1], :]
        random1_score = f(random1_pos.reshape(1, -1))
        random2_score = f(random2_pos.reshape(1, -1))

        new_wolf = quadratic_interpolation(alpha_pos, random1_pos, random2_pos, alpha_score, random1_score, random2_score, lb, ub)
        
        # 處理x1和x2的離散約束，並確保邊界約束
        new_wolf[0, 0] = np.round(new_wolf[0, 0] / 0.0625) * 0.0625
        new_wolf[0, 1] = np.round(new_wolf[0, 1] / 0.0625) * 0.0625
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
            fitness[worst_index] = float(new_wolf_fitness)
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
            print(f"迭代 {t+1}/{max_iter}, 目前最佳適應度: {alpha_score:.4f}")
            print(f"最佳解: Ts={alpha_pos[0]:.4f}, Th={alpha_pos[1]:.4f}, R={alpha_pos[2]:.4f}, L={alpha_pos[3]:.4f}")

    # 打印最終統計信息
    print("\n===== 優化完成 =====")
    print(f"最終最佳適應度(成本): {alpha_score:.4f}")
    print(f"最佳設計參數:")
    print(f"  Ts (殼體厚度) = {alpha_pos[0]:.4f}")
    print(f"  Th (頭部厚度) = {alpha_pos[1]:.4f}")
    print(f"  R (內徑) = {alpha_pos[2]:.4f}")
    print(f"  L (圓柱段長度) = {alpha_pos[3]:.4f}")
    
    # 驗證約束條件
    x = alpha_pos
    g1 = -x[0] + 0.0193 * x[2]
    g2 = -x[1] + 0.00954 * x[2]
    g3 = -np.pi * x[2]**2 * x[3] - (4/3) * np.pi * x[2]**3 + 1296000
    g4 = x[3] - 240
    
    print("\n約束條件檢查:")
    print(f"  g1 = {g1:.6f} {'✓' if g1 <= 0 else '✗'}")
    print(f"  g2 = {g2:.6f} {'✓' if g2 <= 0 else '✗'}")
    print(f"  g3 = {g3:.6f} {'✓' if g3 <= 0 else '✗'}")
    print(f"  g4 = {g4:.6f} {'✓' if g4 <= 0 else '✗'}")
    
    print("\n算法統計:")
    print(f"  二次插值改進次數: {kqc}")
    print(f"  Lévy飛行改進次數: {worstc}")
    print(f"  模擬退火改進次數: {sac}")
    print(f"  逆向學習改進次數: {reverse_fitness_c}")
    print(f"  Fitness計數器: {fitness_counter}")
    print(f"  Fitness計數器OUT: {fitness_counter_OUT}")

    return alpha_pos, alpha_score, best_scores_history

# 主函數
def main():
    # 設置問題參數
    dim = 4  # 設計變數的數量
    # 設計變數的下界和上界
    lb = np.array([0, 0, 10, 10])
    ub = np.array([100, 100, 100, 200])
    
    # 算法參數
    n_wolves = 50  # 狼群數量
    max_iter = 1000  # 最大迭代次數
    
    print("====== 壓力容器設計優化 ======")
    print("設計變數: [Ts, Th, R, L]")
    print(f"變數範圍:")
    print(f"  0 ≤ Ts ≤ 100  (以0.0625為增量)")
    print(f"  0 ≤ Th ≤ 100  (以0.0625為增量)")
    print(f"  10 ≤ R ≤ 100")
    print(f"  10 ≤ L ≤ 200")
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
            pressure_vessel_cost, dim, lb, ub, n_wolves, max_iter
        )
        end_time = time.time()
        execution_time = end_time - start_time
        
        results.append(best_cost)
        best_solutions.append(best_solution)
        execution_times.append(execution_time)
        convergence_histories.append(convergence)
        print(f"運行 {i+1}/30 完成: 最佳成本 = {best_cost:.4f}, 執行時間 = {execution_time:.2f} 秒")
    
    # 計算統計結果
    best_cost_overall = min(results)
    best_index = results.index(best_cost_overall)
    worst_cost_overall = max(results)
    mean_cost = np.mean(results)
    median_cost = np.median(results)
    std_dev = np.std(results)
    mean_execution_time = np.mean(execution_times)
    
    print("\n===== 總統計結果 =====")
    print(f"最佳成本: {best_cost_overall:.4f}")
    print(f"最差成本: {worst_cost_overall:.4f}")
    print(f"平均成本: {mean_cost:.4f}")
    print(f"中位數成本: {median_cost:.4f}")
    print(f"標準差: {std_dev:.4f}")
    print(f"平均執行時間: {mean_execution_time:.2f} 秒")
    
    # 顯示最佳解的詳細信息
    best_sol = best_solutions[best_index]
    print("\n最佳解的詳細信息:")
    print(f"  Ts (殼體厚度) = {best_sol[0]:.4f}")
    print(f"  Th (頭部厚度) = {best_sol[1]:.4f}")
    print(f"  R (內徑) = {best_sol[2]:.4f}")
    print(f"  L (圓柱段長度) = {best_sol[3]:.4f}")
    
    # 驗證最佳解的約束條件
    x = best_sol
    g1 = -x[0] + 0.0193 * x[2]
    g2 = -x[1] + 0.00954 * x[2]
    g3 = -np.pi * x[2]**2 * x[3] - (4/3) * np.pi * x[2]**3 + 1296000
    g4 = x[3] - 240
    
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