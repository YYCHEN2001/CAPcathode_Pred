from sklearn.ensemble import GradientBoostingRegressor

from dataset_function import dataset_load, dataset_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# Initialize the model with Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=900,
                                learning_rate=0.1,
                                max_depth=13,
                                min_samples_leaf=6,
                                min_samples_split=7,
                                subsample=0.8,
                                max_features=0.12,
                                random_state=21)

# Train and evaluate the model
results = train_evaluate(gbr, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, gbr.predict(X_train), y_test, gbr.predict(X_test))

