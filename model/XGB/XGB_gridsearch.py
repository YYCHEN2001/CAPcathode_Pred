import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from dataset_function import data_load, data_split

# Load dataset
X, y = data_load('../../dataset/carbon_20240404.csv')

# Split the dataset
X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.3, random_state=21)

# 设置参数网格
param_grid = {
    # 'learning_rate': np.arange(0.01, 0.31, 0.01),
    'max_depth': np.arange(3, 16, 1),
    'min_child_weight': np.arange(1, 6, 1)
}

# 初始化模型
xgb_model = XGBRegressor(n_estimators=100,
                         # max_depth=12,
                         # min_child_weight=1,
                         learning_rate=0.22,
                         gamma=0.5,
                         subsample=0.2,
                         reg_alpha=0.8,
                         reg_lambda=1,
                         random_state=21)

# 设置交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 初始化GridSearchCV对象
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=kf,
                           scoring='neg_root_mean_squared_error', refit='RMSE')

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 打印最佳参数和对应的分数
print("Best parameters:", grid_search.best_params_)
print("Best RMSE score:", -grid_search.best_score_)
