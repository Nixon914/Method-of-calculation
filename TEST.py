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
    new_pos = best_pos + np.random.uniform(-0.1, 0.1, dim) # 局部扰动
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

# 二次插值函数
def quadratic_interpolation(alpha_pos, beta_pos, delta_pos, alpha_score, beta_score, delta_score):
  X_q, Y_q, Z_q = np.mean(alpha_pos), np.mean(beta_pos), np.mean(delta_pos)
  
  Kq_numerator = (Z_q**2 - Y_q**2) * alpha_score + (X_q**2 - Z_q**2) * beta_score + (Y_q**2 - X_q**2) * delta_score
  Kq_denominator = 2 * ((Z_q - Y_q) * alpha_score + (X_q - Z_q) * beta_score + (Y_q - X_q) * delta_score)
  
  epsilon = 1e-8
  if np.abs(Kq_denominator) < epsilon:
    return X_q
  else:
    K_q = Kq_numerator / Kq_denominator
    return K_q

# GWO-AFSA 主算法，结合模拟退火
def gwo_afsa_with_worst_update(f, dim, lb, ub, n_wolves, max_iter):
  # 初始化种群
  X = np.random.uniform(lb, ub, (n_wolves, dim))

  # 初始化 alpha、beta、delta 狼的位置和分数
  alpha_pos = np.zeros(dim)
  beta_pos = np.zeros(dim)
  delta_pos = np.zeros(dim)
  alpha_score = float('inf')
  beta_score = float('inf')
  delta_score = float('inf')

  # 模拟退火参数
  T0 = 100 # 初始温度
  T_min = 1e-3 # 最低温度
  cooling_rate = 0.9 # 降温速率

  # 迭代过程
  for t in range(max_iter):
    # 计算每只狼的适应度值，更新 alpha、beta、delta
    for i in range(n_wolves):
      fitness = f(X[i, :].reshape(1, -1))
      
      if fitness < alpha_score:
        delta_score, delta_pos = beta_score, beta_pos.copy()
        beta_score, beta_pos = alpha_score, alpha_pos.copy()
        alpha_score, alpha_pos = fitness, X[i, :].copy()
      elif fitness < beta_score:
        delta_score, delta_pos = beta_score, beta_pos.copy()
        beta_score, beta_pos = fitness, X[i, :].copy()
      elif fitness < delta_score:
        delta_score, delta_pos = fitness, X[i, :].copy()
    
    # 计算收敛因子 a
    a = 2 - t * (2 / max_iter)
    
    # 更新每只狼的位置
    for i in range(n_wolves):
      r1 = np.random.rand(dim)
      r2 = np.random.rand(dim)
      A = 2 * a * r1 - a
      C = 2 * r2

      # 更新狼群的位置（根据 alpha、beta、delta）
      if i < n_wolves // 2:
        D_alpha = abs(C * alpha_pos - X[i, :])
        X[i, :] = alpha_pos - A * D_alpha
      else:
        D_beta = abs(C * beta_pos - X[i, :])
        D_delta = abs(C * delta_pos - X[i, :])
        X[i, :] = np.clip((beta_pos - A * D_beta + delta_pos - A * D_delta) / 2, lb, ub)

      # 重新初始化最差的狼
      if i >= n_wolves - 3:
        X[i, :] = np.random.uniform(lb, ub, dim)

    # 使用人工鱼群算法更新最差的狼
    worst_index = np.argmax([f(X[i, :].reshape(1, -1)) for i in range(n_wolves)])
    X[worst_index, :] = afsa_position_update(X[worst_index, :], visual=1, lb=lb, ub=ub)

    # 使用菁英逆向学习生成一个新个体
    X_q = elite_reverse_learning(alpha_pos, lb, ub)
    fitness_q = f(X_q.reshape(1, -1))
    if fitness_q < f(X[worst_index, :].reshape(1, -1)):
      X[worst_index, :] = X_q

    # 使用二次插植生成一个新个体并更新位置
    interpolated_value = quadratic_interpolation(alpha_pos, beta_pos, delta_pos, alpha_score, beta_score, delta_score)
    fitness_interpolated = f(np.full(dim, interpolated_value).reshape(1, -1))
    if fitness_interpolated < alpha_score:
      alpha_pos, alpha_score = np.full(dim, interpolated_value), fitness_interpolated
    
    # 使用模拟退火优化 alpha_pos
    if t % 10 == 0: # 每 10 次迭代进行一次模拟退火
      alpha_pos, alpha_score = simulated_annealing(f, alpha_pos, alpha_score, lb, ub, T0, T_min, cooling_rate)

  return alpha_pos, alpha_score

# 测试 GWO-AFSA 算法
def test_gwo_afsa_on_cec2017():
  dim = 50 # 维度
  n_wolves = 50
  max_iter = 1000
  runs = 30
  lb = -100
  ub = 100
  
  results = []
  
  for i, f in enumerate(functions.all_functions):
    function_results = []
    for run in range(runs):
      best_solution, best_fitness = gwo_afsa_with_worst_update(f, dim, lb, ub, n_wolves, max_iter)
      function_results.append(best_fitness)
      print(f"Function f{i+1}, Run {run + 1}: Best Fitness: {best_fitness}")
    
    results.append(function_results)
  
  return results

# 运行测试
results = test_gwo_afsa_on_cec2017()

