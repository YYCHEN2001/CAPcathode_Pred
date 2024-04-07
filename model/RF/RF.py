from sklearn.ensemble import RandomForestRegressor

from dataset_function import data_load, data_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Load dataset
X, y = data_load('../../dataset/carbon_20240404.csv')

# Split the dataset
X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.2, random_state=21)

# Initialize the model with RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100,
                            max_depth=12,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=21)

# Train and evaluate the model
results = train_evaluate(rfr, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, rfr.predict(X_train), y_test, rfr.predict(X_test))
