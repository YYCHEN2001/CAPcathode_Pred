import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneOut

# 加载数据集
df = pd.read_csv('carbon_cathode_cleaned.csv')  # 请替换为你的文件路径

# 准备数据
X = df.drop('Specific capacity', axis=1)
y = df['Specific capacity']

# 初始化留一交叉验证器
loo = LeaveOneOut()

# 初始化梯度提升回归器
gbr = GradientBoostingRegressor(n_estimators=1300, learning_rate=0.175, max_depth=3,
                                min_samples_leaf=1, min_samples_split=2, random_state=21)

# 存储指标和SHAP值
mae_scores = []
rmse_scores = []
mape_scores = []
shap_values_list = []

# 执行留一交叉验证
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 训练模型
    gbr.fit(X_train, y_train)

    # 预测
    y_pred = gbr.predict(X_test)

    # 计算指标
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape_scores.append(mean_absolute_percentage_error(y_test, y_pred))

    # 生成并存储SHAP值
    explainer = shap.Explainer(gbr)
    shap_values = explainer(X_test)
    shap_values_list.append(shap_values.values)

# 计算平均指标
average_mae = np.mean(mae_scores)
average_rmse = np.mean(rmse_scores)
average_mape = np.mean(mape_scores) * 100

# 合并所有SHAP值用于后续分析
all_shap_values = np.concatenate(shap_values_list, axis=0)

# 绘制SHAP值的摘要图，展示整体分析
shap.summary_plot(all_shap_values, X, plot_type="bar")
shap.summary_plot(all_shap_values, X)
