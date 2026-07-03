import numpy as np
import cec2017.functions as functions

# 模拟退火局部优化函数
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

# 菁英逆向学习
def elite_reverse_learning(X, lb, ub):
    r = np.random.rand(X.shape[0])
    Y = r * (lb + ub) - X
    Y = np.clip(Y, lb, ub)
    return Y

# 人工鱼群算法位置更新
def afsa_position_update(worst_pos, visual, lb, ub):
    rand = np.random.rand()
    new_position = worst_pos + visual * rand
    new_position = np.clip(new_position, lb, ub)
    return new_position

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
            K_q[0, q] = float(numerator / denominator)  # 確保轉換為標量
    
    return K_q



# GWO-AFSA 主算法，结合模拟退火
def gwo_afsa_with_worst_update(f, dim, lb, ub, n_wolves, max_iter):
    # 初始化种群
    X = np.random.uniform(lb, ub, (n_wolves, dim))
    alpha_pos, beta_pos, delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    fitness_counter,kqc,sac,fitness_counter_OUT,worstc,reverse_fitness_c=0,0,0,0,0,0
    # 模拟退火参数
    # 初始温度、最低温度、降温速率
    T0, T_min, cooling_rate = 100, 1e-3, 0.9
    

    for t in range(max_iter):
        #先計算一次適應值
        for i in range(n_wolves):
            fitness = f(X[i, :].reshape(1, -1))
            if np.isinf(fitness):
                fitness = 1e10  # 替代值
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
        #print(f"Alpha: Pos={alpha_pos}, Score={alpha_score}")
        #print(f"Beta: Pos={beta_pos}, Score={beta_score}")
        #print(f"Delta: Pos={delta_pos}, Score={delta_score}")
        # 计算收敛因子 a
        a = 2 - t * (2 / max_iter)

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
            # 步骤 7:菁英算法找逆解
            reverse_position = elite_reverse_learning(alpha_pos, lb, ub)
            reverse_fitness = f(reverse_position.reshape(1, -1))
            # 如果逆向解更好，則更新alpha狼
            if reverse_fitness < alpha_score:
                alpha_score = float(reverse_fitness[0])
                alpha_pos = reverse_position.copy()
                reverse_fitness_c+=1
            else:
                pass
            #步驟8 魚群更新
            # 對適應值進行排序，從小到大
            sorted_indices = np.argsort(fitness)  # 獲取排序後的索引
            worst_indices = sorted_indices[-3:]  # 獲取最差的三隻狼（最後三個索引）
            # 更新最差三隻狼的位置
            for idx in worst_indices:
                new_pos = afsa_position_update(X[idx, :], visual=20, lb=lb, ub=ub)  # 獲取新位置
                new_fitness = f(new_pos.reshape(1, -1))  # 計算新位置的適應值

                # 如果新位置的適應值優於當前位置
                if new_fitness < fitness[idx]:  
                    X[idx, :] = new_pos       # 更新狼的位置
                    fitness[idx] = new_fitness  # 更新適應值
                    worstc += 1               # 計數器 +1，記錄成功更新次數
                
            # 步骤 9更新A、C 
            r1, r2, r3 = np.random.rand(dim), np.random.rand(dim), np.random.rand(dim)
            A1, A2, A3 = 2 * a * r1 - a, 2 * a * r2 - a, 2 * a * r3 - a
            C1, C2, C3 = 2 * np.random.rand(dim), 2 * np.random.rand(dim), 2 * np.random.rand(dim)

            D_alpha = abs(C1 * alpha_pos - X[i, :])
            D_beta = abs(C2 * beta_pos - X[i, :])
            D_delta = abs(C3 * delta_pos - X[i, :])

            #步骤 10根據狼群中每只狼的距離選擇最接近的狼進行位置更新
            if np.min(D_alpha) <= np.min(D_beta) and np.min(D_alpha) <= np.min(D_delta):
                X[i, :] = alpha_pos - A1 * D_alpha
            elif np.min(D_beta) <= np.min(D_alpha) and np.min(D_beta) <= np.min(D_delta):
                X[i, :] = beta_pos - A2 * D_beta
            else:
                X[i, :] = delta_pos - A3 * D_delta

            # 將更新後的位置限制在上下界之內
            X[i, :] = np.clip(X[i, :], lb, ub)

            fitness = np.array([f(X[j, :].reshape(1, -1)) for j in range(n_wolves)])
        
        #步驟14 二次插植

        random_indices = np.random.choice(X.shape[0], 2, replace=False)
        random1_pos = X[random_indices[0], :]
        random2_pos = X[random_indices[1], :]
        random1_score = f(random1_pos.reshape(1, -1))
        random2_score = f(random2_pos.reshape(1, -1))

        new_wolf = quadratic_interpolation(alpha_pos, random1_pos, random2_pos, alpha_score, random1_score, random2_score)
        new_wolf_fitness = f(new_wolf.reshape(1, -1))
        
        # 確保 fitness 的維度正確並更新
        fitness = np.array([f(X[j, :].reshape(1, -1)) for j in range(X.shape[0])])

        # 找到最差狼
        worst_index = np.argmax(fitness)  # 適應值最大的狼即為最差狼
        worst_fitness = fitness[worst_index]
        

        # 如果新狼比最差狼表現更好，進行替換
        if new_wolf_fitness < worst_fitness:
            X[worst_index, :] = new_wolf
            fitness[worst_index] = new_wolf_fitness
            kqc+=1

        if t % 10 == 0:
            Nalpha_score = alpha_score
            alpha_pos, alpha_score = simulated_annealing(f, alpha_pos, alpha_score, lb, ub, T0, T_min, cooling_rate)
            if alpha_score < Nalpha_score:
                sac+=1
    return alpha_score,fitness_counter,kqc,sac,fitness_counter_OUT,worstc,reverse_fitness_c




# 测试 GWO-AFSA 算法
def test_gwo_afsa_on_cec2017():
    dim = 10
    n_wolves = 50
    max_iter = 500
    runs = 1
    lb = -100
    ub = 100

    results = []
    for i, f in enumerate(functions.all_functions):
        function_results = []
        
        # 執行多次測試
        for run in range(runs):
            best_fitness, fitness_counter,kqc,sac,fitness_counter_OUT,worstc,reverse_fitness_c = gwo_afsa_with_worst_update(f, dim, lb, ub, n_wolves, max_iter)
            function_results.append(best_fitness)  # 假設 best_fitness 是數組，取其值
            print(f"Function f{i+1}, Run {run + 1}: Best Fitness: {best_fitness}")
            print(f"fitness_counter{fitness_counter}")
            print(f" kqc{ kqc}")
            print(f"sac{sac}")
            print(f"fitness_counter_OUT{fitness_counter_OUT}")
            print(f"worstc{worstc}")
            print(f"REVERSE{reverse_fitness_c}")
        # 計算統計數據
        best_score = np.min(function_results)
        mean_score = np.average(function_results)
        std_dev = np.std(function_results)
        
        print(f"\nFunction f{i + 1} Statistics:")
        print(f"  Best Fitness: {best_score}")
        print(f"  MEAN Fitness: {mean_score}")
        print(f"  Standard Deviation: {std_dev}")
        print(f"Total fitness evaluations: {fitness_counter}")

    return results

# 运行测试
results = test_gwo_afsa_on_cec2017()

