from sklearn.ensemble import GradientBoostingRegressor
from kfold_cv import perform_kfold_cv
from load_carbon import load

# Load the cleaned dataset
X, y = load('../../dataset/carbon_20240326_2.csv')

# Initialize the model with Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=50,
                                learning_rate=0.26,
                                max_depth=5,
                                min_samples_leaf=8,
                                min_samples_split=5,
                                alpha=0.75,
                                random_state=21)

metrics_df = perform_kfold_cv(gbr, X, y, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(metrics_df)
# Save the DataFrame to a CSV file
metrics_df.to_csv('KFold results of GBR.csv', index=False)
