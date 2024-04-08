from sklearn.ensemble import RandomForestRegressor

from dataset_function import dataset_load, dataset_split
from kfold_cv import perform_kfold_cv

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.2, random_state=21, target='Cs')

# Initialize the model with RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100,
                            max_depth=15,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=21)

# 执行十折交叉验证
metrics_df = perform_kfold_cv(rfr, X_train, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)
# Save the DataFrame to a CSV file
# metrics_df.to_csv('KFold results of RF.csv', index=False)
