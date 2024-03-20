import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor  # Using Gradient Boosting Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneOut

# Load the cleaned dataset
df = pd.read_csv('carbon_20240320.csv')

# Features and Target separation
X = df.iloc[:, :-1]  # Exclude the last column (target)
y = df.iloc[:, -1]  # Target column

# Initialize Leave-One-Out Cross Validator
loo = LeaveOneOut()

# Initialize the model with Gradient Boosting Regression
# n_estimators represents the number of boosting stages to be run.
# learning_rate shrinks the contribution of each tree by learning_rate.
# max_depth is the maximum depth of the individual regression estimators.
gbr = GradientBoostingRegressor(n_estimators=1300, learning_rate=0.175, max_depth=3,
                                min_samples_leaf=1, min_samples_split=2, random_state=21)

# Metrics storage
mae_scores = []
rmse_scores = []
mape_scores = []
train_r2_scores = []

# Perform LOOCV
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model
    gbr.fit(X_train, y_train)

    # Predict
    y_pred = gbr.predict(X_test)

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
