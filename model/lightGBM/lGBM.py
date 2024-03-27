from lightgbm import LGBMRegressor
from load_carbon import load, split_scale
from model_evaluation import train_evaluate, plot_actual_vs_predicted

X, y = load('../../dataset/carbon_20240326.csv')
X_train_scaled, X_test_scaled, y_train, y_test = split_scale(X, y, scale_data=False, test_size=0.3, random_state=21)

# Initialize the model with LightGBM Regression
lgbm = LGBMRegressor(min_child_samples=2,
                     num_leaves=5,
                     max_depth=-1,
                     learning_rate=0.15,
                     n_estimators=1000,
                     random_state=21)
# Train and evaluate the model
results = train_evaluate(lgbm, X_train_scaled, y_train, X_test_scaled, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, lgbm.predict(X_train_scaled), y_test, lgbm.predict(X_test_scaled))
