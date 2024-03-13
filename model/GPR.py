import numpy as np
import pandas as pd
from math import sqrt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold

# 加载数据
data = pd.read_csv('carbon cathode.csv')

# 提取特征和目标值，忽略前三列索引
X = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values

# 定义高斯过程回归模型的内核，包含可能合理的超参数值
kernel = C(1.0, (1e-3, 1e7)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e4))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# 十折交叉验证
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# 评估模型
scores = cross_val_score(gpr, X, y, cv=kfold, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)

# 计算RMSE的平均值和标准差
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

print(f"RMSE: {mean_rmse} (+/- {std_rmse})")
