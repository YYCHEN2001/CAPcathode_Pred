from sklearn.kernel_ridge import KernelRidge

from dataset_function import dataset_load, dataset_split
from kfold_cv import perform_kfold_cv

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.2, random_state=21, target='Cs')

# Initialize the model with Gradient Boosting Regression
krr = KernelRidge(alpha=0.1,
                  gamma=0.1,
                  kernel='polynomial',
                  degree=2,
                  coef0=2.5)

metrics_df = perform_kfold_cv(krr, X_train, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)
