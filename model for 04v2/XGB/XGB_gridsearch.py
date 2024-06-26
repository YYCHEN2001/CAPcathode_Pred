import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from dataset_function import dataset_load, dataset_split

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# 设置参数网格
param_grid = {
    # 'n_estimators': [300, 400, 500],
    # 'learning_rate': np.arange(0.2, 0.3, 0.01),
    # 'max_depth': np.arange(3, 10, 1),
    # 'min_child_weight': np.arange(1, 5, 1),
    # 'subsample': np.arange(0.1, 1, 0.1),
    # 'reg_alpha': np.arange(0.1, 1, 0.1),
    # 'gamma': np.arange(0.1, 1, 0.1),
    'colsample_bytree': np.arange(0.1, 1, 0.1),
    'colsample_bylevel': np.arange(0.1, 1, 0.1),
    'colsample_bynode': np.arange(0.1, 1, 0.1)
}

# 初始化模型
xgb = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    min_child_weight=3,
    learning_rate=0.25,
    gamma=0.1,
    subsample=0.7,
    reg_alpha=0.1,
    # colsample_bytree=0.9,
    # colsample_bylevel=0.4,
    # colsample_bynode=0.75,
    random_state=21)

# 设置交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=21)

# 初始化GridSearchCV对象
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=kf,
                           scoring='neg_mean_absolute_error', refit='MAE',
                           n_jobs=-1, verbose=2)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 打印最佳参数和对应的分数
print("Best parameters:", grid_search.best_params_)
print("Best MAE score:", -grid_search.best_score_)
