from sklearn.kernel_ridge import KernelRidge

from dataset_function import dataset_load, dataset_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404.csv')
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.2, random_state=21, target='Cs')

# Initialize the model with Gradient Boosting Regression
krr = KernelRidge(alpha=0.1,
                  gamma=0.1,
                  kernel='polynomial',
                  degree=2,
                  coef0=2.5)

# Train and evaluate the model
results = train_evaluate(krr, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, krr.predict(X_train), y_test, krr.predict(X_test))
