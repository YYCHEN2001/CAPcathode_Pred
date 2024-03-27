import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load the cleaned dataset
df = pd.read_csv('../../dataset/carbon_20240326_2.csv')

# One-hot encode the categorical columns 'Electrolyte'
df_encoded = pd.get_dummies(df, columns=['Electrolyte'])

# Features and Target separation
X = df_encoded.drop('Cs', axis=1)
y = df_encoded['Cs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# data standard
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.summary_plot(shap_values, X_train)
