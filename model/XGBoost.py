import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# 加载数据集和独热编码
df = pd.read_csv('carbon_20240320.csv')
df_encoded = pd.get_dummies(df, columns=['Electrolyte', 'Current collector'])
X = df_encoded.drop('Cs', axis=1)
y = df_encoded['Cs']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# 初始化XGBoost模型
xgb = XGBRegressor(n_estimators=2000, learning_rate=0.125, max_depth=3, random_state=21)

# 训练模型
xgb.fit(X_train, y_train)

# 预测
y_pred_train = xgb.predict(X_train)
y_pred_test = xgb.predict(X_test)

# 计算和输出训练集和测试集的指标
metrics_train = {
    'R2': r2_score(y_train, y_pred_train),
    'MAE': mean_absolute_error(y_train, y_pred_train),
    'MAPE': mean_absolute_percentage_error(y_train, y_pred_train) * 100,
    'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train))
}

metrics_test = {
    'R2': r2_score(y_test, y_pred_test),
    'MAE': mean_absolute_error(y_test, y_pred_test),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_test) * 100,
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test))
}

print("Training set metrics:", metrics_train)
print("Test set metrics:", metrics_test)

# 十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=21)
cv_results = cross_val_score(xgb, X, y, cv=kf, scoring='r2')

# 输出每一折的R2、MAE、MAPE、RMSE
cv_metrics = []
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    cv_metrics.append({
        'Fold': fold,
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    })

# 输出十折交叉验证的指标
print("Cross-validation metrics per fold:")
for metrics in cv_metrics:
    print(metrics)
