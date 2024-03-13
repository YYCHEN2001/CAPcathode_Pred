
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import ElasticNet  # Using ElasticNet as an example of generalized linear regression
import numpy as np
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv('carbon_cathode_cleaned.csv')

# Features and Target separation
X = df.iloc[:, :-1]  # Exclude the last column (target)
y = df.iloc[:, -1]  # Target column

# Initialize Leave-One-Out Cross Validator
loo = LeaveOneOut()

# Initialize the model with ElasticNet, an example of generalized linear regression
# The l1_ratio is a hyperparameter of ElasticNet. It corresponds to the weight given to the L1 term in the ElasticNet.
# Here, a value of 0.5 means that ElasticNet will have equal weights for L1 and L2 regularization terms.
glr = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=21)

# Metrics storage
mae_scores = []
rmse_scores = []
mape_scores = []

# Perform LOOCV
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model
    glr.fit(X_train, y_train)

    # Predict
    y_pred = glr.predict(X_test)

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
