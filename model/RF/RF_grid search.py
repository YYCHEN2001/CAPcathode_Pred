import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold

from dataset_function import data_load, data_split

# Load dataset
X, y = data_load('../../dataset/carbon_20240404.csv')

# Split the dataset
X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.15, random_state=21)

# 设置参数网格
param_grid = {
    # 'max_depth': np.arange(12, 18, 1),
    # 'min_samples_leaf': np.arange(1, 6, 1),
    'min_samples_split': np.arange(2, 7, 1)
}

# 初始化模型
rfr = RandomForestRegressor(n_estimators=100,
                            max_depth=15,
                            min_samples_leaf=1,
                            random_state=21)

# 设置交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=21)

# 初始化GridSearchCV对象
grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=kf,
                           scoring='neg_root_mean_squared_error', refit='RMSE',
                           n_jobs=-1, verbose=2)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 打印最佳参数和对应的分数
print("Best parameters:", grid_search.best_params_)
print("Best RMSE score:", -grid_search.best_score_)
