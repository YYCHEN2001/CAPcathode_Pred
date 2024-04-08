from lightgbm import LGBMRegressor

from dataset_function import dataset_load, dataset_split
from kfold_cv import perform_kfold_cv

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.2, random_state=21, target='Cs')

# Initialize the model with LightGBM Regression
lgbm = LGBMRegressor(n_estimators=300,
                     max_depth=-1,
                     min_child_samples=2,
                     num_leaves=5,
                     subsample=0.5,
                     learning_rate=0.15,
                     random_state=21)

metrics_df = perform_kfold_cv(lgbm, X_train, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)
# Save the DataFrame to a CSV file
# metrics_df.to_csv('KFold results of LGBM.csv', index=False)
