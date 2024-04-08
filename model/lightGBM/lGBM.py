from lightgbm import LGBMRegressor

from dataset_function import dataset_load, dataset_split
from model_evaluation import train_evaluate, plot_actual_vs_predicted

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')

# Initialize the model with LightGBM Regression
lgbm = LGBMRegressor(n_estimators=300,
                     max_depth=-1,
                     min_child_samples=2,
                     num_leaves=5,
                     subsample=0.5,
                     learning_rate=0.15,
                     random_state=21)
# Train and evaluate the model
results = train_evaluate(lgbm, X_train, y_train, X_test, y_test)
print(results)

# Plot the actual vs predicted values
plot_actual_vs_predicted(y_train, lgbm.predict(X_train), y_test, lgbm.predict(X_test))
