import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# Load the dataset from its location (update the path to where your dataset is stored)
df = pd.read_csv('carbon_cathode_cleaned.csv')

# Calculate the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Draw the heatmap with the mask and correct aspect ratio
heatmap = sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", center=0,
                      square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Pearson相关系数热力图')
plt.tight_layout()

# Save the heatmap to a file (change the file path as needed)
plt.savefig('correlation_heatmap_lower_triangle.png')
plt.show()
