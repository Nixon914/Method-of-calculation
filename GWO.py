import numpy as np

# 定義Rastrigin函數
def rastrigin(X):
    return 10*len(X) + sum([x**2 - 10*np.cos(2*np.pi*x) for x in X])

# 初始化參數
dim = 30
population_size = 30
max_iter = 1000
X = np.random.uniform(-10, 10, (population_size, dim))

# 初始化α狼、β狼和δ狼的位置及適應度
alpha_pos = np.zeros(dim)
beta_pos = np.zeros(dim)
delta_pos = np.zeros(dim)
alpha_score = float('inf')
beta_score = float('inf')
delta_score = float('inf')

# GWO主循環
for l in range(max_iter):
    for i in range(population_size):
        fitness = rastrigin(X[i, :])
        if fitness < alpha_score:
            alpha_score = fitness
            alpha_pos = X[i, :].copy()
        elif fitness < beta_score:
            beta_score = fitness
            beta_pos = X[i, :].copy()
        elif fitness < delta_score:
            delta_score = fitness
            delta_pos = X[i, :].copy()
    
    a = 2 - l * (2 / max_iter)
    
    for i in range(population_size):
        for j in range(dim):
            r1 = np.random.rand()
            r2 = np.random.rand()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha_pos[j] - X[i, j])
            X1 = alpha_pos[j] - A1 * D_alpha
            
            r1 = np.random.rand()
            r2 = np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta_pos[j] - X[i, j])
            X2 = beta_pos[j] - A2 * D_beta
            
            r1 = np.random.rand()
            r2 = np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta_pos[j] - X[i, j])
            X3 = delta_pos[j] - A3 * D_delta
            X[i, j] = (X1 + X2 + X3) / 3

