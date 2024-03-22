import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# Load the datasets
df_train = pd.read_csv('data_75.csv')
df_test = pd.read_csv('data_25.csv')

# Drop the first two columns
df_train = df_train.drop(df_train.columns[[0, 1]], axis=1)
df_test = df_test.drop(df_test.columns[[0, 1]], axis=1)

# Continue with your data preprocessing and model training
# For example, if you need to one-hot encode some columns
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_columns_train = encoder.fit_transform(df_train[['Electrolyte', 'Current collector']]).toarray()
encoded_columns_test = encoder.transform(df_test[['Electrolyte', 'Current collector']]).toarray()
encoded_df_train = pd.DataFrame(encoded_columns_train,
                                columns=encoder.get_feature_names_out(['Electrolyte', 'Current collector']))
encoded_df_test = pd.DataFrame(encoded_columns_test,
                               columns=encoder.get_feature_names_out(['Electrolyte', 'Current collector']))

# Combine data
df_train_encoded = pd.concat([df_train.drop(['Electrolyte', 'Current collector'], axis=1), encoded_df_train], axis=1)
df_test_encoded = pd.concat([df_test.drop(['Electrolyte', 'Current collector'], axis=1), encoded_df_test], axis=1)

# Split the dataset into features and target variable
X_train = df_train_encoded.drop('Cs', axis=1)
y_train = df_train_encoded['Cs']
X_test = df_test_encoded.drop('Cs', axis=1)
y_test = df_test_encoded['Cs']

# 初始化XGBoost模型
xgb = XGBRegressor(n_estimators=5000,  # Increased number of trees
                   max_depth=3,  # Reduced tree depth
                   learning_rate=0.19,  # Lower learning rate
                   reg_alpha=0.75,  # L1 regularization
                   reg_lambda=0.25,
                   n_jobs=-1,
                   random_state=21)

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
