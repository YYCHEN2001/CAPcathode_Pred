import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据
# Load the cleaned dataset
df = pd.read_csv('../dataset/carbon_20240326.csv')

# One-hot encode the categorical columns 'Electrolyte'
df_encoded = pd.get_dummies(df, columns=['Electrolyte'])

# Features and Target separation
X = df_encoded.drop('Cs', axis=1)
y = df_encoded['Cs']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建Lasso回归模型
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # 正则化参数的候选值
lasso_cv = LassoCV(alphas=alphas, cv=5)

# 在训练集上训练模型
lasso_cv.fit(X_train_scaled, y_train)

# 输出选定的最佳alpha值
best_alpha = lasso_cv.alpha_
print("Best alpha:", best_alpha)

# 输出特征选择结果
selected_features = [feature for feature, coef in zip(df_encoded.columns, lasso_cv.coef_) if coef != 0]
print("Selected features:", selected_features)
