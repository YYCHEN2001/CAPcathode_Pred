from xgboost import XGBRegressor
from kfold_cv import perform_kfold_cv
from load_carbon import load

# Load the cleaned dataset
X, y = load('../../dataset/carbon_20240404.csv')

# Initialize the model with XGBoost Regression
xgb = XGBRegressor(n_estimators=200,
                   learning_rate=0.15,
                   max_depth=5,
                   min_child_weight=2,
                   gamma=0.3,
                   subsample=0.2,
                   reg_alpha=0.8,
                   reg_lambda=1,
                   random_state=21)

metrics_df = perform_kfold_cv(xgb, X, y, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)
# Save the DataFrame to a CSV file
metrics_df.to_csv('KFold results of XGB.csv', index=False)
