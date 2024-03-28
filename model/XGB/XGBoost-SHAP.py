import shap
from xgboost import XGBRegressor
from load_carbon import load, split_scale

# Load dataset
X, y = load('../../dataset/carbon_20240326_2.csv')
X_train_scaled, X_test_scaled, y_train, y_test = split_scale(X, y, scale_data=False, test_size=0.3, random_state=21)

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
xgb.fit(X_train_scaled, y_train)
y_pred_train = xgb.predict(X_train_scaled)
y_pred_test = xgb.predict(X_test_scaled)

# Calculate SHAP values
explainer = shap.Explainer(xgb)
shap_values = explainer(X_train_scaled)

# Plot summary of SHAP values
shap.summary_plot(shap_values, X_train_scaled, max_display=8, plot_type="bar", plot_size=(12, 10))
shap.summary_plot(shap_values, X_train_scaled, max_display=8, plot_size=(12, 10))
