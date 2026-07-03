import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import rcParams
import scipy.special as sp

# 設定 matplotlib 字體，避免中文字無法顯示
rcParams['font.sans-serif'] = ['Taipei Sans TC Beta', 'Microsoft JhengHei', 'SimHei']
rcParams['axes.unicode_minus'] = False  # 確保負號正常顯示
#定義壓力容器設計問題的目標函數
def pressure_vessel_cost(x):
    x = x.flatten()
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    #目標函數
    cost = 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3**2 + 3.1661 * x1**2 * x4 + 19.84 * x1**2 * x3
    #處理約束
    penalty = 0
    #約束1
    g1 = -x1 + 0.0193 * x3
    if g1 > 0:
        penalty += 1e6* g1
    #約束2
    g2 = -x2 + 0.00954 * x3
    if g2 > 0:
        penalty += 1e6* g2
    #約束3
    g3 = -np.pi * x3**2 * x4 - (4/3) * np.pi * x3**3 + 1296000
    if g3 > 0:
        penalty += 1e6* g3
    #約束4
    g4 = x4 - 240
    if g4 > 0:
        penalty += 1e15 * g4
    return cost + penalty

def repair_solution(x, lb, ub):
    x = np.clip(x, lb, ub)  # 先確保變數在範圍內
        # 強制修正違反條件的變數
    x[0] = round(x[0] / 0.0625) * 0.0625  # Ts 必須是 0.0625 的倍數
    x[1] = round(x[1] / 0.0625) * 0.0625  # Th 必須是 0.0625 的倍數
    # 強制修正違反條件的變數
    if x[0] > 0.0193 * x[2]:  
        x[0] = 0.0193 * x[2]  # 限制 Ts
    
    if x[1] > 0.00954 * x[2]:  
        x[1] = 0.00954 * x[2]  # 限制 Th
    
    volume = np.pi * x[2]**2 * x[3] + (4/3) * np.pi * x[2]**3
    if volume > 1296000:  
        x[3] = (1296000 - (4/3) * np.pi * x[2]**3) / (np.pi * x[2]**2)  # 限制 L
    
    if x[3] > 240:  
        x[3] = 240  # 限制 L 最大 240
    return x
# 在優化過程中確保 Ts 和 Th 仍然是 0.0625 的倍數
def ensure_discrete_constraints(X):
    X[:, 0] = np.round(X[:, 0] / 0.0625) * 0.0625
    X[:, 1] = np.round(X[:, 1] / 0.0625) * 0.0625
    return X

def simulated_annealing(f, current_pos, current_score, lb, ub, T0, T_min, cooling_rate):
    dim = current_pos.shape[0]
    T = T0
    best_pos = current_pos.copy()
    best_score = current_score

    while T > T_min:
        new_pos = best_pos + np.random.uniform(-0.1, 0.1, dim)
        
        # 對超出邊界的維度進行重置
        for d in range(dim):
            if new_pos[d] < lb[d] or new_pos[d] > ub[d]:
                new_pos[d] = np.random.uniform(lb[d], ub[d])
        
        new_score = f(new_pos.reshape(1, -1))
        
        if new_score < best_score:
            best_pos, best_score = new_pos, new_score
        else:
            acceptance_probability = np.exp(-(new_score - best_score) / T)
            if np.random.rand() < acceptance_probability:
                best_pos, best_score = new_pos, new_score
        T *= cooling_rate
    return best_pos, best_score

def afsa_position_update(worst_pos, visual, lb, ub):
    if worst_pos.ndim == 1:
        worst_pos = worst_pos.reshape(1, -1)
    n_dim = worst_pos.shape[1]

    G = np.zeros_like(worst_pos)
    
    for H in range(n_dim):
        r = np.random.rand()
        new_position = worst_pos[0, H] + visual * r
        
        # 如果超出邊界，重新隨機初始化
        if new_position < lb[H] or new_position > ub[H]:
            G[0, H] = np.random.uniform(lb[H], ub[H])
        else:
            G[0, H] = new_position
    return G

def quadratic_interpolation(alpha_pos, random1_pos, random2_pos, alpha_score, random1_score, random2_score):
    if alpha_pos.ndim == 1:
        alpha_pos = alpha_pos.reshape(1, -1)
    if random1_pos.ndim == 1:
        random1_pos = random1_pos.reshape(1, -1)
    if random2_pos.ndim == 1:
        random2_pos = random2_pos.reshape(1, -1)
    
    n_dim = alpha_pos.shape[1]
    
    K_q = np.zeros((1, n_dim))
    
    for q in range(n_dim):
        X_q = alpha_pos[0, q]
        Y_q = random1_pos[0, q]
        Z_q = random2_pos[0, q]
        
        numerator = (Z_q**2 - Y_q**2) * alpha_score + (X_q**2 - Z_q**2) * random1_score + (Y_q**2 - X_q**2) * random2_score
        
        denominator = 2 * ((Z_q - Y_q) * alpha_score + (X_q - Z_q) * random1_score + (Y_q - X_q) * random2_score)
        
        if abs(denominator) < 1e-10:
            K_q[0, q] = X_q
        else:
            K_q[0, q] = (numerator / denominator).item()
    
    return K_q

def check_improvement(current_score, new_score):
    return new_score < current_score

def levy_flight(dim, sigma=0.01):
    beta = 1.5
    sigma_u = np.power(
        (sp.gamma(1 + beta) * np.sin(np.pi * beta / 2)) /
        (sp.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)),
        1 / beta
    )
    u = np.random.normal(0, sigma_u, dim)
    v = np.random.normal(0, 1, dim)
    step = sigma * u / np.power(np.abs(v), 1/beta)
    return step

def gwo_afsa_with_worst_update(f, dim, lb, ub, n_wolves, max_iter):
    X = np.random.uniform(lb, ub, (n_wolves, dim))
    alpha_pos, beta_pos, delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    fitness_counter,kqc,sac,worstc,fitness_countera,fitness_counterb,fitness_counterc=0,0,0,0,0,0,0
    T0, T_min, cooling_rate = 1000, 1e-2, 0.9
    best_scores_history = []

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)

        fitness_array = np.array([float(f(x.reshape(1, -1))) for x in X])

        fitness_array[np.isinf(fitness_array)] = 1e10
        sorted_indices = np.argsort(fitness_array.flatten())
        alpha_pos = X[sorted_indices[0]].copy()
        beta_pos = X[sorted_indices[1]].copy()
        delta_pos = X[sorted_indices[2]].copy()
        alpha_score = float(fitness_array[sorted_indices[0]])
        beta_score = float(fitness_array[sorted_indices[1]])
        delta_score = float(fitness_array[sorted_indices[2]])
        for i in range(n_wolves):
            fitness = f(X[i, :].reshape(1, -1))
            if np.isinf(fitness):
                fitness = 1e10  # 替代值
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

            sorted_indices = np.argsort(fitness)
            worst_indices = sorted_indices[-5:]

            for idx in worst_indices:
                levy_step = levy_flight(dim)  # 產生獨立的 Levy 變異
                new_pos = X[idx] + levy_step  # 加入 Levy 變異
                new_pos = repair_solution(new_pos, lb, ub)

                for d in range(dim):
                    if new_pos[d] < lb[d] or new_pos[d] > ub[d]:  
                        new_pos[d] = np.random.uniform(lb[d], ub[d])  # 重新初始化該維度

                new_fitness = f(new_pos.reshape(1, -1))

                if check_improvement(fitness_array[idx], new_fitness):
                    X[idx] = new_pos
                    fitness[idx] = new_fitness
                    worstc += 1

            r1, r2, r3 = np.random.rand(dim), np.random.rand(dim), np.random.rand(dim)
            A1, A2, A3 = 2 * a * r1 - a, 2 * a * r2 - a, 2 * a * r3 - a
            C1, C2, C3 = 2 * np.random.rand(dim), 2 * np.random.rand(dim), 2 * np.random.rand(dim)

            D_alpha = abs(C1 * alpha_pos - X[i, :])
            D_beta = abs(C2 * beta_pos - X[i, :])
            D_delta = abs(C3 * delta_pos - X[i, :])

            if np.min(D_alpha) <= np.min(D_beta) and np.min(D_alpha) <= np.min(D_delta):
                X[i, :] = alpha_pos - A1 * D_alpha
            elif np.min(D_beta) <= np.min(D_alpha) and np.min(D_beta) <= np.min(D_delta):
                X[i, :] = beta_pos - A2 * D_beta
            else:
                X[i, :] = delta_pos - A3 * D_delta
            # 對超出邊界的維度進行重置
            for d in range(dim):
                if X[i, d] < lb[d] or X[i, d] > ub[d]:
                    X[i, d] = np.random.uniform(lb[d], ub[d])
            # 计算当前所有狼群的适应值
            fitness_array = np.array([float(f(x.reshape(1, -1))) for x in X])  # 确保是一维数组
            fitness_array[np.isinf(fitness_array)] = 1e10  # 替换无穷值
            # 获取排序索引
            sorted_indices = np.argsort(fitness_array.flatten())
            # 提取最优值对应的解
            new_alpha_pos = X[sorted_indices[0]].copy()
            new_beta_pos = X[sorted_indices[1]].copy()
            new_delta_pos = X[sorted_indices[2]].copy()
            new_alpha_score = float(fitness_array[sorted_indices[0]])
            new_beta_score = float(fitness_array[sorted_indices[1]])
            new_delta_score = float(fitness_array[sorted_indices[2]])
            # 只更新更优的解
            if new_alpha_score < alpha_score:  # 仅在新解优于旧解时更新
                alpha_pos = new_alpha_pos
                alpha_score = new_alpha_score
                fitness_countera += 1
            if new_beta_score < beta_score:
                beta_pos = new_beta_pos
                beta_score = new_beta_score
                fitness_counterb += 1
            if new_delta_score < delta_score:
                delta_pos = new_delta_pos
                delta_score = new_delta_score
                fitness_counterc += 1
                
        random_indices = np.random.choice(X.shape[0], 2, replace=False)
        random1_pos = X[random_indices[0], :]
        random2_pos = X[random_indices[1], :]
        random1_score = f(random1_pos.reshape(1, -1))
        random2_score = f(random2_pos.reshape(1, -1))

        new_wolf = quadratic_interpolation(alpha_pos, random1_pos, random2_pos, alpha_score, random1_score, random2_score)
        new_wolf_fitness = f(new_wolf.reshape(1, -1))

        fitness = np.array([f(X[j, :].reshape(1, -1)) for j in range(X.shape[0])])
        worst_index = np.argmax(fitness)
        worst_fitness = fitness[worst_index]

        if new_wolf_fitness < worst_fitness:
            X[worst_index, :] = new_wolf
            fitness[worst_index] = new_wolf_fitness
            kqc += 1
        if t % 1 == 0:
            Nalpha_score = alpha_score
            alpha_pos, alpha_score = simulated_annealing(f, alpha_pos, alpha_score, lb, ub, T0, T_min, cooling_rate)
            if alpha_score < Nalpha_score:
                sac+=1

        best_scores_history.append(alpha_score)
        
        if (t+1) % 100 == 0 or t == 0:
            print(f"迭代 {t+1}/{max_iter}, 目前最佳適應度: {alpha_score:.4f}")
            print(f"最佳解: Ts={alpha_pos[0]:.4f}, Th={alpha_pos[1]:.4f}, R={alpha_pos[2]:.4f}, L={alpha_pos[3]:.4f}")
        #處理x1和x2的約束
        #處理x1和x2的約束
        X[:, 0] = np.round(X[:, 0] / 0.0625) * 0.0625
        X[:, 1] = np.round(X[:, 1] / 0.0625) * 0.0625
        alpha_pos = X[sorted_indices[0]].copy()
        best_solution = alpha_pos.copy()  # 確保 best_solution 有值
        best_solution = repair_solution(best_solution, lb, ub)

    #打印最終統計信息
    print("\n===== 優化完成 =====")
    print(f"最終最佳適應度(成本): {alpha_score:.4f}")
    print(f"最佳設計參數:")
    print(f"  Ts (殼體厚度) = {alpha_pos[0]:.4f}")
    print(f"  Th (頭部厚度) = {alpha_pos[1]:.4f}")
    print(f"  R (內徑) = {alpha_pos[2]:.4f}")
    print(f"  L (圓柱段長度) = {alpha_pos[3]:.4f}")
    
    #驗證約束條件
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
    print(f"  最差解更新次數: {worstc}")
    print(f"  模擬退火改進次數: {sac}")
    print(f"  Alpha更新次數: {fitness_countera}")
    print(f"  Beta更新次數: {fitness_counterb}")
    print(f"  Delta更新次數: {fitness_counterc}")
    return alpha_pos, alpha_score, best_scores_history

# 主函數
def main():
    # 設置問題參數
    dim = 4  # 設計變數的數量
    lb = np.array([0.0625, 0.0625, 10, 10])  # 設定 Ts 和 Th 的最小值
    ub = np.array([99, 99, 100, 200])
    #算法參數
    
    print("====== 壓力容器設計優化 ======")
    print("設計變數: [Ts, Th, R, L]")
    print(f"變數範圍:")
    print(f"  0 ≤ Ts ≤ 100")
    print(f"  0 ≤ Th ≤ 100")
    print(f"  10 ≤ R ≤ 100")
    print(f"  10 ≤ L ≤ 200")
    print(f"\n開始優化...")
    results = []
    convergence_histories = []
    n_wolves = 100  #狼群數量
    max_iter = 1000  #最大迭代次數
    for i in range(30):
        print(f"\n========== 運行 {i+1}/30 ==========")
        start_time = time.time()
        best_solution, best_cost, convergence = gwo_afsa_with_worst_update(
            pressure_vessel_cost, dim, lb, ub, n_wolves, max_iter
        )
        end_time = time.time()
        execution_time = end_time - start_time
        
        results.append(best_cost)
        convergence_histories.append(convergence)
        print(f"運行 {i+1}/30 完成: 最佳成本 = {best_cost:.4f}, 執行時間 = {execution_time:.2f} 秒")
    
    best_cost_overall = min(results)
    worst_cost_overall = max(results)
    std_dev = np.std(results)
    
    print("\n===== 總統計結果 =====")
    print(f"最佳成本: {best_cost_overall:.4f}")
    print(f"最差成本: {worst_cost_overall:.4f}")
    print(f"標準差: {std_dev:.4f}")
    
    # 繪製所有收斂曲線
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(convergence_histories):
        plt.plot(history, label=f'Run {i+1}')
    plt.title('所有運行的收斂曲線')
    plt.xlabel('迭代次數')
    plt.ylabel('最佳適應度(成本)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return best_solution, best_cost, convergence_histories  # 確保回傳
if __name__ == "__main__":
    best_solution, best_cost, convergence = main()