from xgboost import XGBRegressor
from kfold_cv import perform_kfold_cv
from load_carbon import load

# Load the cleaned dataset
X, y = load('../../dataset/carbon_20240326.csv')
X = X.drop('DV', axis=1)

# Initialize the model with XGBoost Regression
xgb = XGBRegressor(n_estimators=2000,
                   learning_rate=0.14,
                   max_depth=3,
                   min_child_weight=1,
                   gamma=0.5,
                   subsample=0.2,
                   reg_alpha=0.5,
                   reg_lambda=2,
                   random_state=21)

metrics_df = perform_kfold_cv(xgb, X, y, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)
# Save the DataFrame to a CSV file
metrics_df.to_csv('KFold results of XGB_drop_DV.csv', index=False)
