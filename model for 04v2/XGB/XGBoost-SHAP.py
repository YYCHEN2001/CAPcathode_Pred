import shap
from xgboost import XGBRegressor

from dataset_function import dataset_load, dataset_split

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')

X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.2, random_state=21, target='Cs')

# Initialize the model with XGBoost Regression
xgb = XGBRegressor(n_estimators=100,
                   learning_rate=0.2,
                   max_depth=9,
                   min_child_weight=2,
                   gamma=0.2,
                   subsample=0.4,
                   reg_alpha=0.8,
                   reg_lambda=1,
                   random_state=21)
xgb.fit(X_train, y_train)
y_pred_train = xgb.predict(X_train)
y_pred_test = xgb.predict(X_test)

# Calculate SHAP values
explainer = shap.Explainer(xgb)
shap_values = explainer(X_train)

# Plot summary of SHAP values
shap.summary_plot(shap_values, X_train, max_display=0, plot_type="bar", plot_size=(12, 10))
shap.summary_plot(shap_values, X_train, max_display=0, plot_size=(12, 10))
