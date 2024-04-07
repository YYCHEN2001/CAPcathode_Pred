from lightgbm import LGBMRegressor

from dataset_function import data_load, data_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Load dataset
X, y = data_load('../../dataset/carbon_20240404.csv')

# Split the dataset
X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.2, random_state=21)

# Initialize the model with LightGBM Regression
lgbm = LGBMRegressor(min_child_samples=2,
                     num_leaves=5,
                     max_depth=-1,
                     learning_rate=0.15,
                     n_estimators=1000,
                     random_state=21)
# Train and evaluate the model
results = train_evaluate(lgbm, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, lgbm.predict(X_train), y_test, lgbm.predict(X_test))
