from xgboost import XGBRegressor

from pdp_function import (load_split_data, combine_data,
                          predict_plot)

filename = '../../../dataset/carbon_20240326_2.csv'
X_train, X_test, y_train, y_test, base_features = load_split_data(filename, 'Cs', test_size=0.3, random_state=21)
train_df, test_df = combine_data(X_train, X_test, y_train, y_test, base_features)

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

fig, summary_df = predict_plot(xgb, test_df, base_features, 'DV', 'Potential_window', 'matplotlib')
fig.show()
