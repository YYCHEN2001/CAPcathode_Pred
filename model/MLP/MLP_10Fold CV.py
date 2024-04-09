from sklearn.neural_network import MLPRegressor

from dataset_function import dataset_load, dataset_split
from kfold_cv import perform_kfold_cv

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# Initialize the model with MLP Regression
mlp = MLPRegressor(hidden_layer_sizes=(50, 50, 50),
                   activation='relu',
                   solver='adam',
                   alpha=0.0001,
                   batch_size='auto',
                   learning_rate='constant',
                   learning_rate_init=0.001,
                   max_iter=200,
                   shuffle=True,
                   random_state=21)

metrics_df = perform_kfold_cv(mlp, X_train, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)
