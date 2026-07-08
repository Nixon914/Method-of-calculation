import numpy as np
import cec2017.functions as functions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fitness_counter = 0
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
def logistic_map(x, r=3.99):
    return r * x * (1 - x)




# GWO-AFSA 主算法，结合模拟退火
def gwo_afsa_with_worst_update(f, dim, lb, ub, n_wolves, max_iter):
    # 初始化种群
    X = np.random.uniform(lb, ub, (n_wolves, dim))
    alpha_pos, beta_pos, delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    fitness_counter,kqc,sac,fitness_counter_OUT,worstc=0,0,0,0,0
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
            fitness = np.array([f(x.reshape(1, -1)) for x in X])
            worst_indices = np.argsort(fitness)[-5:]  # 找到适应度最差的5个个体

            for idx in worst_indices:
                levy_step = levy_flight(dim)
                new_pos = X[idx] + levy_step
                new_pos = np.clip(new_pos, lb, ub)  # 边界限制
                new_fitness = f(new_pos.reshape(1, -1))
                
            
                
                if new_fitness < fitness[idx]:  # 若新解优于当前解
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

            #步骤 10根據狼群中每只狼的距離選擇最接近的狼進行位置更新
            if np.min(D_alpha) <= np.min(D_beta) and np.min(D_alpha) <= np.min(D_delta):
                X[i, :] = alpha_pos - A1 * D_alpha
            elif np.min(D_beta) <= np.min(D_alpha) and np.min(D_beta) <= np.min(D_delta):
                X[i, :] = beta_pos - A2 * D_beta
            else:
                X[i, :] = delta_pos - A3 * D_delta

            # 將更新後的位置限制在上下界之內
            X[i, :] = np.clip(X[i, :], lb, ub)

            fitness = np.array([f(X[l, :].reshape(1, -1)) for l in range(n_wolves)])
        
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
    return alpha_score,fitness_counter,kqc,sac,fitness_counter_OUT,worstc




# 测试 GWO-AFSA 算法
def test_gwo_afsa_on_cec2017():
    dim = 10
    n_wolves = 50
    max_iter = 1000
    runs = 30
    lb = -100
    ub = 100

    results = []
    for i, f in enumerate(functions.all_functions):
        function_results = []
        
        # 執行多次測試
        for run in range(runs):
            best_fitness, fitness_counter,kqc,sac,fitness_counter_OUT,worstc = gwo_afsa_with_worst_update(f, dim, lb, ub, n_wolves, max_iter)
            function_results.append(best_fitness)  # 假設 best_fitness 是數組，取其值
            print(f"Function f{i+1}, Run {run + 1}: Best Fitness: {best_fitness}")
            print(f"fitness_counter{fitness_counter}")
            print(f" kqc{ kqc}")
            print(f"sac{sac}")
            print(f"fitness_counter_OUT{fitness_counter_OUT}")
            print(f"worstc{worstc}")
        # 計算統計數據
        best_score = np.min(function_results)
        worst_score = np.max(function_results)
        std_dev = np.std(function_results)
        
        print(f"\nFunction f{i + 1} Statistics:")
        print(f"  Best Fitness: {best_score}")
        print(f"  Worst Fitness: {worst_score}")
        print(f"  Standard Deviation: {std_dev}")
        print(f"Total fitness evaluations: {fitness_counter}")

        
        # 保存結果以便後續使用
        results.append({
            'function': f"f{i + 1}",
            'best': best_score,
            'worst': worst_score,
            'std_dev': std_dev
        })

    return results
