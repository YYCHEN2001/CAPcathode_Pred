import shap
from xgboost import XGBRegressor

from dataset_function import data_load, data_split

# Load dataset
X, y = data_load('../../dataset/carbon_20240404.csv')

# Split the dataset
X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.15, random_state=21)

# Initialize the model with XGBoost Regression
xgb = XGBRegressor(n_estimators=2000,
                   learning_rate=0.15,
                   max_depth=3,
                   min_child_weight=1,
                   gamma=0.5,
                   subsample=0.2,
                   reg_alpha=0.5,
                   reg_lambda=2,
                   random_state=21)
xgb.fit(X_train, y_train)
y_pred_train = xgb.predict(X_train)
y_pred_test = xgb.predict(X_test)

# Calculate SHAP values
explainer = shap.Explainer(xgb)
shap_values = explainer(X_train)

# Plot summary of SHAP values
shap.summary_plot(shap_values, X_train, max_display=8, plot_type="bar", plot_size=(12, 10))
shap.summary_plot(shap_values, X_train, max_display=8, plot_size=(12, 10))
