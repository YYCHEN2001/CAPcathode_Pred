import shap
from sklearn.ensemble import GradientBoostingRegressor

from dataset_function import dataset_load, dataset_split

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')


# Initialize the model with XGBoost Regression
gbr = GradientBoostingRegressor(n_estimators=900,
                                learning_rate=0.1,
                                max_depth=13,
                                min_samples_leaf=6,
                                min_samples_split=7,
                                subsample=0.8,
                                max_features=0.12,
                                random_state=21)
gbr.fit(X_train, y_train)
y_pred_train = gbr.predict(X_train)
y_pred_test = gbr.predict(X_test)

# Calculate SHAP values
explainer = shap.Explainer(gbr)
shap_values = explainer(X_train)

# Plot summary of SHAP values
shap.summary_plot(shap_values, X_train, plot_type="bar", plot_size=(12, 10))
shap.summary_plot(shap_values, X_train, plot_size=(21, 10))
