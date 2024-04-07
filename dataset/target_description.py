import matplotlib.pyplot as plt
import pandas as pd

from dataset_function import feature_normalize

# Load the dataset
df = pd.read_csv('carbon_20240404.csv')

# Drop specific columns
df_filtered = df.drop(['Current collector', 'Electrolyte'], axis=1)

# Normalize the features
normalized_array = feature_normalize(df_filtered)

# Convert the normalized numpy array back to a pandas DataFrame
# Use the column names from the filtered DataFrame
df_normalized = pd.DataFrame(normalized_array, columns=df_filtered.columns)

# Descriptive statistics and missing values
descriptive_stats = df_normalized.describe()
missing_values = df_normalized.isnull().sum()

# Print descriptive statistics and missing values
print(descriptive_stats)
print("\nMissing values for each column:\n", missing_values)

# Plotting boxplots for all numerical features
plt.figure(figsize=(10, 6))
df_normalized.boxplot()
plt.xticks(rotation=45)
plt.title("Boxplots of numerical features excluding 'Current collector' and 'Electrolyte'")
plt.tight_layout()
plt.show()
