import numpy as np
def simulated_annealing(f, current_pos, current_score, lb, ub, T0, T_min, cooling_rate):
    dim = current_pos.shape[0]
    T = T0
    best_pos = current_pos.copy()
    best_score = current_score

    while T > T_min:
        # 生成一个新的邻域解
        new_pos = best_pos + np.random.uniform(-0.1, 0.1, dim)  # 局部扰动
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
def elite_reverse_learning(alpha_pos, lb, ub):
    # 確保輸入是二維數組
    if alpha_pos.ndim == 1:
        alpha_pos = alpha_pos.reshape(1, -1)
    n_dim = alpha_pos.shape[1]  # 使用shape[1]獲取維度
    #初始化Y（逆解的更新）
    Y = np.zeros_like(alpha_pos)
    
    for K in range(n_dim):
        r = np.random.rand()
        # 计算逆解，确保它是标量
        inverse_solution = r * (lb + ub) - alpha_pos[0, K]
        # 更新Y中的第K维度的逆解
        Y[0,K] = inverse_solution
        # 如果逆解超出边界，则随机初始化一个新的解
        if Y[0,K] < lb or Y[0,K] > ub:
            Y[0,K] = np.random.uniform(lb, ub)# 在边界范围内随机生成新解
    
    return Y

# 人工鱼群算法位置更新
def afsa_position_update(worst_pos, visual, lb, ub):
    # 確保輸入是二維數組
    if worst_pos.ndim == 1:
        worst_pos = worst_pos.reshape(1, -1)
    n_dim = worst_pos.shape[1]
    
    # 初始化G
    G = np.zeros_like(worst_pos)
    
    # 對每個維度計算新位置
    for H in range(n_dim):
        r = np.random.rand()
        new_position = worst_pos[0,H] + visual * r
        
        # 修改這裡：檢查邊界並單獨處理每個維度
        if isinstance(lb, (int, float)):
            # 如果lb是單一數值
            if new_position < lb or new_position > ub[H]:
                G[0,H] = np.random.uniform(lb, ub[H])
            else:
                G[0,H] = new_position
        else:
            # 如果lb是陣列
            if new_position < lb[H] or new_position > ub[H]:
                G[0,H] = np.random.uniform(lb[H], ub[H])
            else:
                G[0,H] = new_position
    
    return G
#二次插植
def quadratic_interpolation(alpha_pos, random1_pos, random2_pos, alpha_score, random1_score, random2_score):

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
        X_q = alpha_pos[0, q]  # 注意索引變化
        Y_q = random1_pos[0, q]
        Z_q = random2_pos[0, q]
        
        # 計算分子
        numerator = (Z_q**2 - Y_q**2) * alpha_score + (X_q**2 - Z_q**2) * random1_score + (Y_q**2 - X_q**2) * random2_score
        
        # 計算分母
        denominator = 2 * ((Z_q - Y_q) * alpha_score + (X_q - Z_q) * random1_score + (Y_q - X_q) * random2_score)
        
        # 避免除零
        if abs(denominator) < 1e-10:
            K_q[0, q] = X_q
        else:
            K_q[0, q] = (numerator / denominator).item()  # 確保轉換為標量
    
    return K_q



# GWO-AFSA 主算法，结合模拟退火
def gwo_afsa_with_worst_update(f, dim, lb, ub, n_wolves, max_iter):
    # 初始化种群
    X = np.random.uniform(lb, ub, (n_wolves, dim))
    alpha_pos, beta_pos, delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    kqc,sac,worstc,fitness_countera,fitness_counterb,fitness_counterc=0,0,0,0,0,0
    T0, T_min, cooling_rate = 100, 1e-3, 0.9
    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)
        # 修改适应度计算方式
        

        # 正确获取排序索引
        # 更新Alpha、Beta、Delta
        # 第一次迭代時用排序設定初始值
        if t == 0:
            fitness_array = np.array([float(f(x.reshape(1, -1))[0]) for x in X])  # 确保是一维数组
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
            sorted_indices = np.argsort(fitness)
            worst_indices = sorted_indices[-5:]
            for idx in worst_indices:
                new_pos = afsa_position_update(X[idx, :], visual=20, lb=lb, ub=ub)
                new_fitness = f(new_pos.reshape(1, -1))
                if new_fitness < fitness[idx]:
                    X[idx, :] = new_pos
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

            X[i, :] = np.clip(X[i, :], lb, ub)
            # 计算当前所有狼群的适应值
            fitness_array = np.array([float(f(x.reshape(1, -1))[0]) for x in X])  # 确保是一维数组
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
        if t % 10 == 0:
            Nalpha_score = alpha_score
            alpha_pos, alpha_score = simulated_annealing(f, alpha_pos, alpha_score, lb, ub, T0, T_min, cooling_rate)
            if alpha_score < Nalpha_score:
                sac+=1
    print(f"Debug: alpha_score = {alpha_score}")  # 检查 alpha_score 类型
    return alpha_score,kqc, worstc,sac,fitness_countera,fitness_counterb,fitness_counterc
# 定義問題的目標函數和約束條件
def vessel_objective(x):
    """
    返回一個包含目標函數值和約束違反程度的數組
    x 需要是 (1, 4) 的形狀
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # 目標函數
    f = 0.6224*x[0,0]*x[0,2]*x[0,3] + 1.7781*x[0,1]*x[0,2]**2 + 3.1661*x[0,0]**2*x[0,3] + 19.84*x[0,0]**2*x[0,2]
    
    # 約束條件違反程度
    g1 = max(0, -x[0,0] + 0.0193*x[0,2])
    g2 = max(0, -x[0,1] + 0.00954*x[0,2])
    g3 = max(0, -(np.pi * x[0,2]**2 * x[0,3]) - (4/3 * np.pi * x[0,2]**3) + 1296000)
    g4 = max(0, x[0,3] - 240)
    
    # 將約束違反加入到目標函數中（懲罰法）
    penalty = 1e6  # 懲罰係數
    f_penalized = f 
    
    return np.array([f_penalized])

# 設定問題參數
dim = 4  # 4個設計變數
n_wolves = 30  # 狼群大小
max_iter = 500  # 最大迭代次數

# 設定變數邊界
lb = 0  # 下界
ub = np.array([100, 100, 100, 200])  # 上界

# 執行算法
best_score, kqc, worstc, sac, fa, fb, fc = gwo_afsa_with_worst_update(
    vessel_objective, 
    dim, 
    lb, 
    ub, 
    n_wolves, 
    max_iter
)

print("\n最終結果:")
print(f"最佳目標函數值: {best_score}")
print(f"最差解更新次數: {worstc}")
print(f"Alpha位置更新次數: {fa}")
#print(f"g1=: {g1}")