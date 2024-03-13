
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.kernel_ridge import KernelRidge  # Using Kernel Ridge Regression
import numpy as np
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv('carbon_cathode_cleaned.csv')

# Features and Target separation
X = df.iloc[:, :-1]  # Exclude the last column (target)
y = df.iloc[:, -1]  # Target column

# Initialize Leave-One-Out Cross Validator
loo = LeaveOneOut()

# Initialize the model with Kernel Ridge Regression
# Here, alpha is a regularization parameter (similar to the lambda in ridge regression).
# The 'kernel' parameter specifies the kernel type to be used in the algorithm.
# 'rbf' is the Radial Basis Function kernel, a common choice for kernel ridge regression.
krr = KernelRidge(alpha=1.0, kernel='rbf')

# Metrics storage
mae_scores = []
rmse_scores = []
mape_scores = []

# Perform LOOCV
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model
    krr.fit(X_train, y_train)

    # Predict
    y_pred = krr.predict(X_test)

    # Metrics calculation
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape_scores.append(mean_absolute_percentage_error(y_test, y_pred))

# Average metrics
average_mae = np.mean(mae_scores)
average_rmse = np.mean(rmse_scores)
average_mape = np.mean(mape_scores) * 100  # Convert to percentage

print(f'Average MAE: {average_mae}')
print(f'Average RMSE: {average_rmse}')
print(f'Average MAPE: {average_mape}%')
