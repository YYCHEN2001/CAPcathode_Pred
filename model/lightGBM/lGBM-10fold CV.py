from lightgbm import LGBMRegressor

from dataset_function import data_load
from kfold_cv import perform_kfold_cv

# Load the cleaned dataset
X, y = data_load('../../dataset/carbon_20240404.csv')

# Initialize the model with LightGBM Regression
lgbm = LGBMRegressor(min_child_samples=2,
                     num_leaves=5,
                     max_depth=-1,
                     learning_rate=0.15,
                     n_estimators=1000,
                     random_state=21)

metrics_df = perform_kfold_cv(lgbm, X, y, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)
# Save the DataFrame to a CSV file
metrics_df.to_csv('KFold results of LGBM.csv', index=False)
