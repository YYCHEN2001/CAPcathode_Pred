from sklearn.ensemble import GradientBoostingRegressor

from dataset_function_for_v3 import dataset_load, dataset_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v3.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# Initialize the model with Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=200,
                                learning_rate=0.1,
                                max_depth=8,
                                min_samples_leaf=1,
                                min_samples_split=8,
                                subsample=0.75,
                                max_features=0.25,
                                random_state=21)

# Train and evaluate the model
results = train_evaluate(gbr, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, gbr.predict(X_train), y_test, gbr.predict(X_test))
