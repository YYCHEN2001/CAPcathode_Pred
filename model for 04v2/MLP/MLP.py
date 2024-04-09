from sklearn.neural_network import MLPRegressor

from dataset_function import dataset_load, dataset_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# Initialize the model with MLP Regression
mlp = MLPRegressor(hidden_layer_sizes=(50, 50, 50),
                   activation='relu',
                   solver='adam',
                   alpha=0.01,
                   batch_size='auto',
                   learning_rate='constant',
                   learning_rate_init=0.01,
                   max_iter=10000,
                   shuffle=True,
                   random_state=21)

# Train and evaluate the model
results = train_evaluate(mlp, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, mlp.predict(X_train), y_test, mlp.predict(X_test))
