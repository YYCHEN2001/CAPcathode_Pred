from sklearn.ensemble import RandomForestRegressor
from kfold_cv import perform_kfold_cv

# Load the cleaned dataset
from load_carbon import load
X, y = load('../../dataset/carbon_20240326.csv')

# Initialize the model with RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=50,
                            max_depth=12,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=21)

# 执行十折交叉验证
metrics_df = perform_kfold_cv(rfr, X, y, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)
# Save the DataFrame to a CSV file
metrics_df.to_csv('KFold results of RF.csv', index=False)
