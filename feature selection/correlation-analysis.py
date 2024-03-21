import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the dataset from its location (update the path to where your dataset is stored)
df = pd.read_csv('carbon_20240320.csv')

# Calculate the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(24, 24))

# Adjust font size
font_size = 24  # You can adjust this value according to your needs
plt.rcParams['font.sans-serif'] = ['SimHei']  # Set font to support Chinese characters if needed
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = font_size  # Set the general font size

# Draw the heatmap with the mask and correct aspect ratio
heatmap = sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", center=0,
                      square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": font_size})

plt.title('Pearson相关系数热力图', fontsize=font_size + 12)  # Slightly larger font size for the title
plt.tight_layout()

# Save the heatmap to a file (change the file path as needed)
plt.savefig('correlation_heatmap_lower_triangle.png')
plt.show()
