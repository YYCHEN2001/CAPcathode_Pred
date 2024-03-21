import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
data_path = 'carbon_20240320.csv'
data = pd.read_csv(data_path)

# 计算斯皮尔曼等级相关系数
spearman_corr = data.corr(method='spearman')

# 创建遮罩以仅显示热力图的下三角部分
mask = np.triu(np.ones_like(spearman_corr, dtype=bool))

# 绘制斯皮尔曼等级相关系数的下三角热力图
plt.figure(figsize=(12, 12))
sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Lower Triangle of Spearman Rank Correlation Coefficients Heatmap")
plt.show()
