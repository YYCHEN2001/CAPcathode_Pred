import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, \
    root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load the cleaned dataset
df = pd.read_csv('../../dataset/carbon_20240326_2.csv')

# One-hot encode the categorical columns 'Electrolyte'
df_encoded = pd.get_dummies(df, columns=['Electrolyte'])

# Features and Target separation
X = df_encoded.drop('Cs', axis=1)
y = df_encoded['Cs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# data standard
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model with XGBoost Regression
xgb = XGBRegressor(n_estimators=2000,
                   learning_rate=0.15,
                   max_depth=3,
                   min_child_weight=1,
                   gamma=0.5,
                   subsample=0.1,
                   reg_alpha=0.5,
                   reg_lambda=2,
                   random_state=21)
xgb.fit(X_train_scaled, y_train)
y_pred_train = xgb.predict(X_train_scaled)
y_pred_test = xgb.predict(X_test_scaled)


def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return r2, mae, mape, rmse


metrics_train = calculate_metrics(y_train, y_pred_train)
metrics_test = calculate_metrics(y_test, y_pred_test)

results = pd.DataFrame({
    'Metric': ['R2', 'MAE', 'MAPE', 'RMSE'],
    'train set': metrics_train,
    'Test set': metrics_test
})

print(results)

# Plot the actual vs predicted values
plt.figure(figsize=(10, 10))
plt.scatter(y_train, y_pred_train, color='blue', label='Train')
plt.scatter(y_test, y_pred_test, color='red', label='Test')

# Fit a line to the points
z = np.polyfit(y_train, y_pred_train, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "k--", label='Fit line')

plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()
