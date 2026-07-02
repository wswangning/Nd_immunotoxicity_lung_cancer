from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np

# 定义参数范围
problem = {
    'num_vars': 4,
    'names': ['air_conc', 'part_coeff', 'oral_bioavail', 'oral_nd_frac'],
    'bounds': [[0.02, 0.08], [0.003, 0.015], [0.0005, 0.002], [0.10, 0.20]]
}

# 生成样本
param_values = saltelli.sample(problem, 1024)

# 运行模型（需要编写一个函数，输入参数，返回MOE）
Y = np.zeros(param_values.shape[0])
for i, p in enumerate(param_values):
    Y[i] = compute_moe(p[0], p[1], p[2], p[3])   # 注意compute_moe需实际模拟

# 分析
Si = sobol.analyze(problem, Y)
print(Si['S1'])   # 一阶敏感度