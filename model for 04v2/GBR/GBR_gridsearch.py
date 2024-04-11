import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold

from dataset_function import dataset_load, dataset_split

# Load dataset
df = dataset_load('../../dataset/carbon_202404_v2.csv')

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# 设置参数网格
param_grid = {
    'n_estimators': np.arange(10, 300, 10),
    'learning_rate': np.arange(0.16, 0.2, 0.01),
    'max_depth': np.arange(3, 8, 1),
    'min_samples_leaf': np.arange(1, 4, 1),
    'min_samples_split': np.arange(2, 8, 1),
    # 'alpha': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
    # 'subsample': np.arange(0.1, 1, 0.1),
    # 'max_features': np.arange(0.1, 1, 0.1),
}

# 初始化模型
gbr = GradientBoostingRegressor(n_estimators=200,
                                learning_rate=0.17,
                                max_depth=4,
                                min_samples_leaf=1,
                                min_samples_split=2,
                                alpha=0.001,
                                subsample=0.8,
                                max_features=0.2,
                                random_state=21)

# 设置交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=21)

# 初始化GridSearchCV对象
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=kf,
                           scoring='neg_mean_absolute_error', refit='MAE',
                           n_jobs=-1, verbose=2)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 打印最佳参数和对应的分数
print("Best parameters:", grid_search.best_params_)
print("Best MAE score:", -grid_search.best_score_)
