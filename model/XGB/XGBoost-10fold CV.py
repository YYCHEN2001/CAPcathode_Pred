from xgboost import XGBRegressor

from dataset_function import data_load, data_split
from kfold_cv import perform_kfold_cv

# Load the cleaned dataset
X, y = data_load('../../dataset/carbon_20240404.csv')

# Split the dataset
X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.15, random_state=21)

# Initialize the model with XGBoost Regression
xgb = XGBRegressor(n_estimators=100,
                   learning_rate=0.2,
                   max_depth=8,
                   min_child_weight=1,
                   gamma=0.5,
                   subsample=0.2,
                   reg_alpha=0.8,
                   reg_lambda=1,
                   random_state=21)

metrics_df = perform_kfold_cv(xgb, X, y, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)
# Save the DataFrame to a CSV file
metrics_df.to_csv('KFold results of XGB.csv', index=False)
