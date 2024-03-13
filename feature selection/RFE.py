import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Import datasets
data = pd.read_csv('carbon cathode.csv')
# Setting the first three columns as indexes and separating features and target
X = data.iloc[:, 3:-1]  # Features
y = data.iloc[:, -1]  # Target
# Splitting the dataset into training and testing sets with a 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# Initialize the estimator
estimator = LinearRegression()

# Loop for n_features_to_select from 3 to 14
results = []  # to store the selected features for each number of features to select
for n_features in range(1, 15):
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