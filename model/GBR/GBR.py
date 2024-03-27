from sklearn.ensemble import GradientBoostingRegressor
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Load the cleaned dataset
from load_carbon import load, split_scale
X, y = load('../../dataset/carbon_20240326.csv')
X_train_scaled, X_test_scaled, y_train, y_test = split_scale(X, y, scale_data=False, test_size=0.3, random_state=21)

# Initialize the model with Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=2000,
                                learning_rate=0.1,
                                max_depth=3,
                                min_samples_leaf=8,
                                min_samples_split=5,
                                alpha=0.75,
                                random_state=21)

# Train and evaluate the model
results = train_evaluate(gbr, X_train_scaled, y_train, X_test_scaled, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, gbr.predict(X_train_scaled), y_test, gbr.predict(X_test_scaled))
