from xgboost import XGBRegressor

from dataset_function import data_load, data_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Load dataset
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

# Train and evaluate the model
results = train_evaluate(xgb, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, xgb.predict(X_train_scaled), y_test, xgb.predict(X_test_scaled))
