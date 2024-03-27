import pymrmr
import pandas as pd

# Assuming the cleaned dataset is loaded
df = pd.read_csv('../dataset/carbon_20240326.csv')

# Separating features and target without combining them for mRMR
X = df.iloc[:, :-1]  # All columns except the last one are features
y = df.iloc[:, -1]   # The last column is the target

# mRMR expects the target to be part of the DataFrame but not as a feature to select.
# So, we correctly add it as a separate column for clarity and compliance with mRMR requirements.
df_mrmr = pd.concat([X, y.rename('Target')], axis=1)

# Loop over a range of n_features from 1 to 15
results = []  # Store the selected features for each number of features
for n_features in range(1, 16):
    # Use mRMR to select the top N features, ensuring 'Target' is not among the features to be selected but used as the variable to guide selection
    selected_features = pymrmr.mRMR(df_mrmr, 'MIQ', n_features)

    # Append the results
    results.append((n_features, selected_features))

# Displaying results more appropriately
for n_features, features in results:
    print(f"Top {n_features} features selected by mRMR: {features}")
