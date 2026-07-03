import numpy as np
import matplotlib.pyplot as plt
import time

def speed_reducer_design_cost(x):
    """
    減速器設計問題的目標函數和約束條件
    x1: 齒輪面寬(b)
    x2: 齒輪模數(m)
    x3: 小齒輪齒數(z)
    x4: 第一軸承長度(L1)
    x5: 第二軸承長度(L2)
    x6: 第一軸直徑(d1)
    x7: 第二軸直徑(d2)
    """
    x = x.flatten()
    x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    
    # 目標函數 - 減速器重量
    f = 0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934) - 1.508 * x1 * (x6**2 + x7**2) + 7.4777 * (x6**3 + x7**3) + (x4 * x6**2 + x5 * x7**2)
    
    # 處理約束
    penalty = 0
    
    # 約束1
    g1 = 27 / (x1 * x2**2 * x3) - 1
    if g1 > 0:
        penalty += 1e6 * g1
    
    # 約束2
    g2 = 397.5 / (x1 * x2**2 * x3**2) - 1
    if g2 > 0:
        penalty += 1e6 * g2
    
    # 約束3
    g3 = 1.93 * x4**3 / (x2 * x6**4 * x3) - 1
    if g3 > 0:
        penalty += 1e6 * g3
    
    # 約束4
    g4 = 1.93 * x5**3 / (x2 * x7**4 * x3) - 1
    if g4 > 0:
        penalty += 1e6 * g4
    
    # 約束5
    g5 = np.sqrt((745 * x4 / (x2 * x3))**2 + 16.9e6) / (110 * x6**3) - 1
    if g5 > 0:
        penalty += 1e6 * g5
    
    # 約束6
    g6 = np.sqrt((745 * x5 / (x2 * x3))**2 + 157.5e6) / (85 * x7**3) - 1
    if g6 > 0:
        penalty += 1e6 * g6
    
    # 約束7
    g7 = x2 * x3 / 40 - 1
    if g7 > 0:
        penalty += 1e6 * g7
    
    # 約束8
    g8 = 5 * x2 / x1 - 1
    if g8 > 0:
        penalty += 1e6 * g8
    
    # 約束9
    g9 = x1 / (12 * x2) - 1
    if g9 > 0:
        penalty += 1e6 * g9
    
    # 約束10
    g10 = (1.5 * x6 + 1.9) / x4 - 1
    if g10 > 0:
        penalty += 1e6 * g10
    
    # 約束11
    g11 = (1.1 * x7 + 1.9) / x5 - 1
    if g11 > 0:
        penalty += 1e6 * g11
    
    return f + penalty

def simulated_annealing(f, current_pos, current_score, lb, ub, T0, T_min, cooling_rate):
    dim = current_pos.shape[0]
    T = T0
    best_pos = current_pos.copy()
    best_score = current_score

    while T > T_min:
        new_pos = best_pos + np.random.uniform(-0.1, 0.1, dim)
        new_pos = np.clip(new_pos, lb, ub)
        
        new_score = f(new_pos.reshape(1, -1))
        
        if new_score < best_score:
            best_pos, best_score = new_pos, new_score
        else:
            acceptance_probability = np.exp(-(new_score - best_score) / T)
            if np.random.rand() < acceptance_probability:
                best_pos, best_score = new_pos, new_score
        T *= cooling_rate
    return best_pos, best_score

def elite_reverse_learning(alpha_pos, lb, ub):
    if alpha_pos.ndim == 1:
        alpha_pos = alpha_pos.reshape(1, -1)
    n_dim = alpha_pos.shape[1]
    Y = np.zeros_like(alpha_pos)
    
    for K in range(n_dim):
        r = np.random.rand()
        inverse_solution = r * (lb[K] + ub[K]) - alpha_pos[0, K]
        Y[0, K] = inverse_solution
        if Y[0, K] < lb[K] or Y[0, K] > ub[K]:
            Y[0, K] = np.random.uniform(lb[K], ub[K])
    return Y

def afsa_position_update(worst_pos, visual, lb, ub):
    if worst_pos.ndim == 1:
        worst_pos = worst_pos.reshape(1, -1)
    n_dim = worst_pos.shape[1]

    G = np.zeros_like(worst_pos)
    
    for H in range(n_dim):
        r = np.random.rand()
        new_position = worst_pos[0, H] + visual * r
        
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

def gwo_afsa_with_worst_update(f, dim, lb, ub, n_wolves, max_iter):
    # 初始化狼群
    X = np.random.uniform(lb, ub, (n_wolves, dim))
    
    alpha_pos, beta_pos, delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    kqc, sac, worstc, fitness_countera, fitness_counterb, fitness_counterc = 0, 0, 0, 0, 0, 0
    T0, T_min, cooling_rate = 100, 1e-3, 0.9
    
    best_scores_history = []
    
    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)
        
        if t == 0:
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
            
            # 確保所有變數在邊界內
            X[i, :] = np.clip(X[i, :], lb, ub)
        
        fitness = np.array([f(X[j, :].reshape(1, -1)) for j in range(X.shape[0])])
        sorted_indices = np.argsort(fitness)
        worst_indices = sorted_indices[-5:]
        
        for idx in worst_indices:
            new_pos = afsa_position_update(X[idx, :], visual=0.2, lb=lb, ub=ub)
            
            new_fitness = f(new_pos.reshape(1, -1))
            if new_fitness < fitness[idx]:
                X[idx, :] = new_pos.flatten()
                fitness[idx] = new_fitness
                worstc += 1
        
        fitness_array = np.array([f(x.reshape(1, -1)) for x in X])
        fitness_array[np.isinf(fitness_array)] = 1e10

        sorted_indices = np.argsort(fitness_array.flatten())

        new_alpha_pos = X[sorted_indices[0]].copy()
        new_beta_pos = X[sorted_indices[1]].copy()
        new_delta_pos = X[sorted_indices[2]].copy()
        new_alpha_score = float(fitness_array[sorted_indices[0]])
        new_beta_score = float(fitness_array[sorted_indices[1]])
        new_delta_score = float(fitness_array[sorted_indices[2]])

        if new_alpha_score < alpha_score:
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

        worst_index = np.argmax(fitness)
        worst_fitness = fitness[worst_index]

        if new_wolf_fitness < worst_fitness:
            X[worst_index, :] = new_wolf.flatten()
            fitness[worst_index] = new_wolf_fitness
            kqc += 1
        
        if t % 10 == 0:
            Nalpha_score = alpha_score
            alpha_pos, alpha_score = simulated_annealing(f, alpha_pos, alpha_score, lb, ub, T0, T_min, cooling_rate)
            
            if alpha_score < Nalpha_score:
                sac += 1

        best_scores_history.append(alpha_score)
        
        if (t+1) % 100 == 0 or t == 0:
            print(f"迭代 {t+1}/{max_iter}, 目前最佳適應度: {alpha_score:.4f}")
            print(f"最佳解: b={alpha_pos[0]:.4f}, m={alpha_pos[1]:.4f}, z={alpha_pos[2]:.4f}, L1={alpha_pos[3]:.4f}, L2={alpha_pos[4]:.4f}, d1={alpha_pos[5]:.4f}, d2={alpha_pos[6]:.4f}")
    
    # 打印最終統計信息
    print("\n===== 優化完成 =====")
    print(f"最終最佳適應度(成本): {alpha_score:.6f}")
    print(f"最佳設計參數:")
    print(f"  b (齒輪面寬) = {alpha_pos[0]:.6f}")
    print(f"  m (齒輪模數) = {alpha_pos[1]:.6f}")
    print(f"  z (小齒輪齒數) = {alpha_pos[2]:.6f}")
    print(f"  L1 (第一軸承長度) = {alpha_pos[3]:.6f}")
    print(f"  L2 (第二軸承長度) = {alpha_pos[4]:.6f}")
    print(f"  d1 (第一軸直徑) = {alpha_pos[5]:.6f}")
    print(f"  d2 (第二軸直徑) = {alpha_pos[6]:.6f}")
    
    # 驗證約束條件
    x = alpha_pos
    x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    
    # 計算約束函數值
    g1 = 27 / (x1 * x2**2 * x3) - 1
    g2 = 397.5 / (x1 * x2**2 * x3**2) - 1
    g3 = 1.93 * x4**3 / (x2 * x6**4 * x3) - 1
    g4 = 1.93 * x5**3 / (x2 * x7**4 * x3) - 1
    g5 = np.sqrt((745 * x4 / (x2 * x3))**2 + 16.9e6) / (110 * x6**3) - 1
    g6 = np.sqrt((745 * x5 / (x2 * x3))**2 + 157.5e6) / (85 * x7**3) - 1
    g7 = x2 * x3 / 40 - 1
    g8 = 5 * x2 / x1 - 1
    g9 = x1 / (12 * x2) - 1
    g10 = (1.5 * x6 + 1.9) / x4 - 1
    g11 = (1.1 * x7 + 1.9) / x5 - 1
    
    print("\n約束條件檢查:")
    print(f"  g1 = {g1:.6f} {'✓' if g1 <= 0 else '✗'}")
    print(f"  g2 = {g2:.6f} {'✓' if g2 <= 0 else '✗'}")
    print(f"  g3 = {g3:.6f} {'✓' if g3 <= 0 else '✗'}")
    print(f"  g4 = {g4:.6f} {'✓' if g4 <= 0 else '✗'}")
    print(f"  g5 = {g5:.6f} {'✓' if g5 <= 0 else '✗'}")
    print(f"  g6 = {g6:.6f} {'✓' if g6 <= 0 else '✗'}")
    print(f"  g7 = {g7:.6f} {'✓' if g7 <= 0 else '✗'}")
    print(f"  g8 = {g8:.6f} {'✓' if g8 <= 0 else '✗'}")
    print(f"  g9 = {g9:.6f} {'✓' if g9 <= 0 else '✗'}")
    print(f"  g10 = {g10:.6f} {'✓' if g10 <= 0 else '✗'}")
    print(f"  g11 = {g11:.6f} {'✓' if g11 <= 0 else '✗'}")
    
    # 已知文獻中的最佳解
    best_known = np.array([3.6, 0.8, 17, 7.3, 7.8, 3.9, 5.5])
    best_known_cost = 0.7854 * best_known[0] * best_known[1]**2 * (3.3333 * best_known[2]**2 + 14.9334 * best_known[2] - 43.0934) - 1.508 * best_known[0] * (best_known[5]**2 + best_known[6]**2) + 7.4777 * (best_known[5]**3 + best_known[6]**3) + (best_known[3] * best_known[5]**2 + best_known[4] * best_known[6]**2)
    
    print("\n文獻中的最佳解:")
    print(f"  b={best_known[0]:.1f}, m={best_known[1]:.1f}, z={best_known[2]:.1f}, L1={best_known[3]:.1f}, L2={best_known[4]:.1f}, d1={best_known[5]:.1f}, d2={best_known[6]:.1f}")
    print(f"  成本 = {best_known_cost:.6f}")
    
    # 計算兩者差異百分比
    cost_diff_percent = abs(alpha_score - best_known_cost) / best_known_cost * 100
    print(f"\n與文獻最佳解的差異: {cost_diff_percent:.4f}%")
    
    print("\n算法統計:")
    print(f"  二次插值改進次數: {kqc}")
    print(f"  最差解更新次數: {worstc}")
    print(f"  模擬退火改進次數: {sac}")
    print(f"  Alpha更新次數: {fitness_countera}")
    print(f"  Beta更新次數: {fitness_counterb}")
    print(f"  Delta更新次數: {fitness_counterc}")
    
    # 繪製收斂曲線
    plt.figure(figsize=(10, 6))
    plt.plot(best_scores_history, 'b-', linewidth=2)
    plt.title('優化收斂曲線')
    plt.xlabel('迭代次數')
    plt.ylabel('最佳適應度(成本)')
    plt.grid(True)
    plt.show()
    
    return alpha_pos, alpha_score, best_scores_history

# 主函數
def main():
    # 設置問題參數
    dim = 7  # 設計變數的數量
    # 設計變數的下界和上界
    lb = np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0])
    ub = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])
    
    # 算法參數
    n_wolves = 50  # 狼群數量
    max_iter = 500  # 最大迭代次數
    
    print("減速器設計優化問題")
    print("設計變數: [b,m,z,L1,L2,d1,d2]")
    print(f"\n開始優化...")
    
    # 記錄開始時間
    start_time = time.time()
    
    # 執行優化
    best_solution, best_cost, convergence = gwo_afsa_with_worst_update(
        speed_reducer_design_cost, dim, lb, ub, n_wolves, max_iter
    )
    
    # 計算執行時間
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n總執行時間: {execution_time:.2f} 秒")
    
    return best_solution, best_cost, convergence

if __name__ == "__main__":
    best_solution, best_cost, convergence = main()