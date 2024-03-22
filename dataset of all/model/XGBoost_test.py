import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# Load the datasets
df_train = pd.read_csv('dataset_train.csv')
df_test = pd.read_csv('dataset_test.csv')

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
X_train = df_train_encoded.drop('Specific capacity', axis=1)
y_train = df_train_encoded['Specific capacity']
X_test = df_test_encoded.drop('Specific capacity', axis=1)
y_test = df_test_encoded['Specific capacity']

# 训练XGBoost模型
xgb_model = XGBRegressor(n_estimators=3000,  # Increased number of trees
                         max_depth=4,  # Reduced tree depth
                         learning_rate=0.05,  # Lower learning rate
                         reg_alpha=0.5,  # L1 regularization
                         reg_lambda=0.25,
                         n_jobs=-1,
                         random_state=21)
xgb_model.fit(X_train, y_train)

# 进行预测
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# 计算指标
metrics = {
    'R2': (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)),
    'MAE': (mean_absolute_error(y_train, y_train_pred), mean_absolute_error(y_test, y_test_pred)),
    'MAPE': (
        mean_absolute_percentage_error(y_train, y_train_pred),
        mean_absolute_percentage_error(y_test, y_test_pred)
    ),
    'RMSE': (
        root_mean_squared_error(y_train, y_train_pred),
        root_mean_squared_error(y_test, y_test_pred)
    )
}

# 将结果以DataFrame形式输出
results_df = pd.DataFrame(metrics, index=['Train', 'Test'])
print(results_df)
