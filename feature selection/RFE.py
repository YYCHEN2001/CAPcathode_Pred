import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv('../dataset/carbon_20240320.csv')

# One-hot encode the categorical columns 'Electrolyte' and 'Current collector'
df_encoded = pd.get_dummies(df, columns=['Electrolyte', 'Current collector'])

# Features and Target separation
X = df_encoded.drop('Cs', axis=1)
y = df_encoded['Cs']

# Splitting the dataset into training and testing sets with an 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Initialize the estimator
estimator = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.125, max_depth=3,
                                min_samples_leaf=1, min_samples_split=2, random_state=21)

# Loop for n_features_to_select from 1 to 19
results = []  # to store the selected features for each number of features to select
for n_features in range(1, 20):
    # Instantiate RFE with the current number of features to select
    selector = RFE(estimator, n_features_to_select=n_features)

    # Fit RFE to the data
    selector.fit(X_train, y_train)

    # Get the selected features
    selected_features = X_train.columns[selector.support_]

    # Append the result
    results.append((n_features, selected_features.tolist()))

# Print the results
for n_features, features in results:
    print(f"Selected {n_features} features: {features}")