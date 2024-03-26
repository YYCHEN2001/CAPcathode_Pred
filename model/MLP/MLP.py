import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, \
    root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('carbon_20240326_2.csv')
df_encoded = pd.get_dummies(df, columns=['Electrolyte'])
X = df_encoded.drop('Cs', axis=1)
y = df_encoded['Cs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(50, 50, 50),
                   activation='relu',
                   solver='lbfgs',
                   max_iter=10000,
                   alpha=0.001,
                   learning_rate_init=0.001,
                   random_state=21)
mlp.fit(X_train_scaled, y_train)
y_pred_train = mlp.predict(X_train_scaled)
y_pred_test = mlp.predict(X_test_scaled)

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return r2, mae, mape, rmse


metrics_train = calculate_metrics(y_train, y_pred_train)
metrics_test = calculate_metrics(y_test, y_pred_test)

results = pd.DataFrame({
    'Metric': ['R2', 'MAE', 'MAPE', 'RMSE'],
    'train set': metrics_train,
    'Test set': metrics_test
})

print(results)

# Plot the actual vs predicted values
plt.figure(figsize=(10, 10))
plt.scatter(y_train, y_pred_train, color='blue', label='Train')
plt.scatter(y_test, y_pred_test, color='red', label='Test')

# Fit a line to the points
z = np.polyfit(y_train, y_pred_train, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "k--", label='Fit line')

plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()




