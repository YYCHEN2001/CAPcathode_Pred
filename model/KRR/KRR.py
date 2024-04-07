from sklearn.kernel_ridge import KernelRidge

from dataset_function import data_load, data_split, feature_normalize
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Load dataset
X, y = data_load('../../dataset/carbon_20240404.csv')

# Normalize the features
X_normalized = feature_normalize(X)

# Split the dataset
X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.2, random_state=21)

# Initialize the model with Gradient Boosting Regression
krr = KernelRidge(alpha=0.01,
                  gamma=0.1,
                  kernel='polynomial',
                  degree=2,
                  coef0=2.5)

# Train and evaluate the model
results = train_evaluate(krr, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, krr.predict(X_train), y_test, krr.predict(X_test))
