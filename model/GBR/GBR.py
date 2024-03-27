import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, \
    root_mean_squared_error

# Load the cleaned dataset
from read_carbon import load_and_process_data

X, y, X_train_scaled, X_test_scaled, y_train, y_test = load_and_process_data('../../dataset/carbon_20240326.csv')

# Initialize the model with Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=2000,
                                learning_rate=0.1,
                                max_depth=3,
                                min_samples_leaf=8,
                                min_samples_split=5,
                                alpha=0.75,
                                random_state=21)
gbr.fit(X_train_scaled, y_train)
y_pred_train = gbr.predict(X_train_scaled)
y_pred_test = gbr.predict(X_test_scaled)


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
