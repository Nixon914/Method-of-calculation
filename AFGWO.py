import numpy as np
import cec2017.functions as functions

def objective_function(X):
    return 10*len(X) + sum([x**2 - 10*np.cos(2*np.pi*x) for x in X])
#初始化參數
dim = 30
population_size = 30
max_iter = 1000
X = np.random.uniform(-10, 10, (population_size, dim))
#初始化α狼、β狼和δ狼的位置及適應度
alpha_pos = np.zeros(dim)
beta_pos = np.zeros(dim)
delta_pos = np.zeros(dim)
alpha_score = float('inf')
beta_score = float('inf')
delta_score = float('inf')
#二次插值
def K_q(alpha_pos, beta_pos, delta_pos, alpha_score, beta_score, delta_score):
    numerator = ((delta_pos ** 2 - beta_pos ** 2) * alpha_score + (alpha_pos ** 2 - delta_pos ** 2) * beta_score + (beta_pos ** 2 - alpha_pos ** 2) * delta_score)
    denominator = 2 * ((delta_pos - beta_pos) * alpha_score + (alpha_pos - delta_pos) * beta_score + (beta_pos - alpha_pos) * delta_score)
    
    if denominator != 0:
        K_q = numerator / denominator
    else:
        raise ValueError("Denominator cannot be zero.")

    return K_q
#AFGWO主循環
for t in range(max_iter):
    for i in range(population_size):
        fitness = objective_function(X[i, :])
        if fitness < alpha_score:
            alpha_score = fitness
            alpha_pos = X[i, :].copy()
        elif fitness < beta_score:
            beta_score = fitness
            beta_pos = X[i, :].copy()
        elif fitness < delta_score:
            delta_score = fitness
            delta_pos = X[i, :].copy()
    
    a = 2 - t * (2 / max_iter)
    
    for i in range(population_size):
        if i < 3:
            X[i, :] = np.clip(-X[i, :] + (X[i, :] + 10), -10, 10)
        r1 = np.random.rand(dim)
        r2 = np.random.rand(dim)
        A = 2 * a * r1 - a
        C = 2 * r2
        D_alpha = abs(C * alpha_pos - X[i, :])
        X1 = alpha_pos - A * D_alpha

        D_beta = abs(C * beta_pos - X[i, :])
        X2 = beta_pos - A * D_beta

        D_delta = abs(C * delta_pos - X[i, :])
        X3 = delta_pos - A * D_delta

        X[i, :] = (X1 + X2 + X3) / 3
        
        if i >= population_size - 3:  
            X[i, :] = np.random.uniform(-10, 10, dim)
    #精英逆向學習
    if t % 10 == 0:
        new_individual = quadratic_interpolation(alpha_pos, beta_pos, delta_pos, 
                                                 objective_function(alpha_pos), 
                                                 objective_function(beta_pos), 
                                                 objective_function(delta_pos))
        X[np.random.randint(0, population_size), :] = np.clip(new_individual, -10, 10)


