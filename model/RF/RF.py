from sklearn.ensemble import RandomForestRegressor
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Load dataset
from load_carbon import load, split_scale
X, y = load('../../dataset/carbon_20240326.csv')
X_train_scaled, X_test_scaled, y_train, y_test = split_scale(X, y, scale_data=False, test_size=0.3, random_state=21)

# Initialize the model with RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=2000,
                            max_depth=9,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=21)

# Train and evaluate the model
results = train_evaluate(rfr, X_train_scaled, y_train, X_test_scaled, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, rfr.predict(X_train_scaled), y_test, rfr.predict(X_test_scaled))
