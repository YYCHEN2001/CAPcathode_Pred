import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 加载数据集
df = pd.read_csv('carbon_cathode_cleaned.csv')

# 定义分类特征和数值特征
categorical_features = ['Electrolyte', 'Current collector']
numerical_features = df.drop(['Specific capacity', 'Electrolyte', 'Current collector'], axis=1).columns.tolist()

# 数据预处理：独热编码分类特征
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 初始化梯度提升回归器
gbr = GradientBoostingRegressor(n_estimators=1300, learning_rate=0.175, max_depth=3,
                                min_samples_leaf=1, min_samples_split=2, random_state=21)

# 创建处理和模型训练的管道
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', gbr)])

# 准备数据
X = df.drop('Specific capacity', axis=1)
y = df['Specific capacity']

# 训练模型
pipeline.fit(X, y)

# SHAP解释器应该直接作用于模型，而非整个管道
explainer = shap.Explainer(pipeline.named_steps['regressor'])

# 获取预处理后的数据
X_transformed = pipeline.named_steps['preprocessor'].transform(X)

# 使用预处理后的数据生成SHAP值
shap_values = explainer.shap_values(X_transformed)

# 绘制SHAP值的摘要图
shap.summary_plot(shap_values, X_transformed)
