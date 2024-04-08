import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from dataset_function import dataset_load, dataset_split

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.2, random_state=21, target='Cs')

# 设置参数网格
param_grid = {
    # 'learning_rate': np.arange(0.05, 0.45, 0.05),
    # 'max_depth': np.arange(5, 11, 1),
    # 'min_child_weight': np.arange(1, 6, 1)
    'gamma': np.arange(0, 1, 0.1),
    'subsample': np.arange(0.1, 1, 0.1),
    'reg_alpha': np.arange(0, 1, 0.1),
    'reg_lambda': np.arange(0, 1, 0.1)
}

# 初始化模型
xgb = XGBRegressor(n_estimators=100,
                   max_depth=9,
                   min_child_weight=2,
                   learning_rate=0.2,
                   # gamma=0.5,
                   # subsample=0.2,
                   # reg_alpha=0.8,
                   # reg_lambda=1,
                   random_state=21)

# 设置交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 初始化GridSearchCV对象
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=kf,
                           scoring='neg_root_mean_squared_error', refit='RMSE',
                           n_jobs=-1, verbose=2)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 打印最佳参数和对应的分数
print("Best parameters:", grid_search.best_params_)
print("Best RMSE score:", -grid_search.best_score_)
