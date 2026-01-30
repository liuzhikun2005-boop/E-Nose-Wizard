import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载训练集和测试集数据
train_df = pd.read_csv('train_dataclass.csv')  # 假设训练集数据保存在 train_data.csv 文件中
test_df = pd.read_csv('test_dataclass.csv')    # 假设测试集数据保存在 test_data.csv 文件中

# 提取训练集特征和标签
X_train = train_df.iloc[:, :-1].values  # 特征
y_train = train_df.iloc[:, -1].values   # 标签

# 提取测试集特征和标签
X_test = test_df.iloc[:, :-1].values    # 特征
y_test = test_df.iloc[:, -1].values     # 标签

# 数据归一化
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# 使用 PCA 进行降维至二维
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_normalized)
X_test_pca = pca.transform(X_test_normalized)

# 构建 SVM 模型
svm_model = SVC(probability=True)

# 训练 SVM 模型
svm_model.fit(X_train_pca, y_train)

# 绘制 PCA 可视化界面
plt.figure()
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.colorbar(label='Class')
plt.show()
