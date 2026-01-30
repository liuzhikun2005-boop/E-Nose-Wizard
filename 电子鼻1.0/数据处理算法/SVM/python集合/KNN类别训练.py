import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 读取训练集数据和验证集数据
train_data = pd.read_csv("train_dataclass.csv")  # 替换为训练集数据文件名
test_data = pd.read_csv("test_dataclass.csv")  # 替换为验证集数据文件名

# 分离特征和标签
X_train = train_data.iloc[:, :4]
y_train = train_data.iloc[:, 4]
X_test = test_data.iloc[:, :4]
y_true = test_data.iloc[:, 4]

# 归一化处理，将特征数据缩放到 [0, 1]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化列表以存储不同k值下的准确率
k_values = list(range(1, 11))
accuracies = []

# 遍历不同的k值
for k in k_values:
    # 训练KNN分类器
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    # 预测验证集数据
    y_pred = knn.predict(X_test_scaled)

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    accuracies.append(accuracy)

# 初始化列表以存储不同k值下的准确率（未进行归一化）
accuracies_unscaled = []

# 遍历不同的k值
for k in k_values:
    # 训练KNN分类器（未进行归一化）
    knn_unscaled = KNeighborsClassifier(n_neighbors=k)
    knn_unscaled.fit(X_train, y_train)

    # 预测验证集数据（未进行归一化）
    y_pred_unscaled = knn_unscaled.predict(X_test)

    # 计算准确率（未进行归一化）
    accuracy_unscaled = accuracy_score(y_true, y_pred_unscaled)
    accuracies_unscaled.append(accuracy_unscaled)

# 可视化结果
plt.plot(k_values, accuracies, marker='o', label='With Scaling')
plt.plot(k_values, accuracies_unscaled, marker='o', label='Without Scaling')
plt.title('Accuracy vs. Number of Neighbors (K)')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()
