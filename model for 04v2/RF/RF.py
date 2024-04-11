from sklearn.ensemble import RandomForestRegressor

from dataset_function import dataset_load, dataset_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# Initialize the model with RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=200,
                            max_depth=15,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=21)

# Train and evaluate the model
results = train_evaluate(rfr, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, rfr.predict(X_train), y_test, rfr.predict(X_test))
