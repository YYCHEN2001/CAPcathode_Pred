from load_carbon import load, split_scale
from xgboost import XGBRegressor
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Load dataset
X, y = load('../../dataset/carbon_20240326.csv')
X_train_scaled, X_test_scaled, y_train, y_test = split_scale(X, y, scale_data=False, test_size=0.3, random_state=21)

# Initialize the model with XGBoost Regression
xgb = XGBRegressor(n_estimators=2000,
                   learning_rate=0.15,
                   max_depth=3,
                   min_child_weight=1,
                   gamma=0.5,
                   subsample=0.2,
                   reg_alpha=0.5,
                   reg_lambda=2,
                   random_state=21)

# Train and evaluate the model
results = train_evaluate(xgb, X_train_scaled, y_train, X_test_scaled, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, xgb.predict(X_train_scaled), y_test, xgb.predict(X_test_scaled))