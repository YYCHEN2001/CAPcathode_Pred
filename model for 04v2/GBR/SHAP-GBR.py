import shap
from sklearn.ensemble import GradientBoostingRegressor

from dataset_function import dataset_load, dataset_split

# Split the dataset into training and testing sets, using quantile-based stratification for the target variable.
df = dataset_load('../../dataset/carbon_202404_v2.csv')
X_train, X_test, y_train, y_test = dataset_split(df, test_size=0.3, random_state=21, target='Cs')


# Initialize the model with XGBoost Regression
gbr = GradientBoostingRegressor(n_estimators=200,
                                learning_rate=0.17,
                                max_depth=4,
                                min_samples_leaf=1,
                                min_samples_split=2,
                                alpha=0.001,
                                subsample=0.8,
                                max_features=0.2,
                                random_state=21)
gbr.fit(X_train, y_train)
y_pred_train = gbr.predict(X_train)
y_pred_test = gbr.predict(X_test)

# Calculate SHAP values
explainer = shap.Explainer(gbr)
shap_values = explainer(X_test)

# Plot summary of SHAP values
shap.summary_plot(shap_values, X_test, plot_type="bar", plot_size=(12, 10))
shap.summary_plot(shap_values, X_test, plot_size=(25.6, 14.4))

shap.initjs()  # 初始化用于显示Jupyter notebook中的JS可视化的环境
force_plot = shap.force_plot(explainer.expected_value, shap_values.values[0, :], X_test.iloc[0, :])
shap.save_html('force_plot.html', force_plot)
