import pymrmr
import pandas as pd

# Load your data
df = pd.read_csv('carbon cathode.csv')

# Assuming the last column is the target and the rest are features
X = df.iloc[:, 3:-1]  # Adjust based on your dataset
y = df.iloc[:, -1]

# Combine features and target for mRMR
df_mrmr = pd.concat([X, y.rename('Target')], axis=1)

# Loop over a range of n_features from 1 to 14
results = []  # Store the selected features for each number of features
for n_features in range(1, 15):
    # Use mRMR to select the top N features
    selected_features = pymrmr.mRMR(df_mrmr, 'MIQ', n_features)

    # Append the results
    results.append((n_features, selected_features))

# Print the results
for n_features, features in results:
    print(f"Top {n_features} features selected by mRMR: {features}")