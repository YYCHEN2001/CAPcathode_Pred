import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('carbon_20240320.csv')  # 请替换为您的数据集文件路径

# Separate features; assuming the last column is the target
X = data.iloc[:, :-1]

# Initialize the matrix to store mutual information values
n_features = X.shape[1]
mi_matrix_continuous = np.zeros((n_features, n_features))

# Calculate mutual information for each pair of features
for i in range(n_features):
    for j in range(i + 1, n_features):
        # We treat one of the features as the target in each pair
        mi = mutual_info_regression(X.iloc[:, [i]], X.iloc[:, j])[0]
        mi_matrix_continuous[i, j] = mi
        mi_matrix_continuous[j, i] = mi  # Fill the symmetric matrix

# Convert the matrix to a DataFrame for better readability
mi_matrix_continuous_df = pd.DataFrame(mi_matrix_continuous, columns=X.columns, index=X.columns)

# Round the matrix to 3 decimal places
mi_matrix_continuous_df_rounded = mi_matrix_continuous_df.round(3)

# Display the lower triangle of the matrix
mi_matrix_continuous_lower_triangle = mi_matrix_continuous_df_rounded.where(
    np.tril(np.ones(mi_matrix_continuous_df.shape), k=-1).astype(bool)
)

print(mi_matrix_continuous_lower_triangle)

# Plot the lower triangle of the mutual information matrix with a color map where higher values are darker
plt.figure(figsize=(12, 10))
sns.heatmap(mi_matrix_continuous_lower_triangle, annot=True, fmt=".3f", cmap="Blues",
            cbar_kws={'label': 'Mutual Information'}, mask=mi_matrix_continuous_lower_triangle.isnull())
plt.title("Mutual Information between Features")
plt.show()