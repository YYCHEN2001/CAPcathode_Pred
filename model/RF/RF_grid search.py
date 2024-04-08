import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold

from dataset_function import dataset_load, dataset_split

# Load dataset
df = dataset_load('../../dataset/carbon_202404_v2.csv')

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.2, random_state=21, target='Cs')

# 设置参数网格
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': np.arange(8, 12, 1),
    'min_samples_leaf': np.arange(1, 4, 1),
    'min_samples_split': np.arange(2, 5, 1)
}

# 初始化模型
rfr = RandomForestRegressor(
    # n_estimators=100,
    # min_samples_leaf=1,
    # min_samples_split=2,
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
