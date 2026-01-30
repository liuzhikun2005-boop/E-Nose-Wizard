import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from perfume_classfication import model

# 加载数据集D:\python集合\svm(pytorch).py
data = pd.read_csv("dataclass.csv")

# 提取特征和标签
features = data.iloc[:, :-1]  # 前四列是特征
labels = data.iloc[:, -1]     # 最后一列是标签
# 读取数据集
data = pd.read_csv("dataclass.csv")  # 替换为你的数据集路径



# 归一化处理
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

# 将归一化后的特征和标签重新组合成DataFrame
normalized_data = pd.DataFrame(normalized_features, columns=features.columns)
normalized_data['label'] = labels

# 保存处理后的数据集到新的CSV文件
normalized_data.to_csv("normalized_perfume_data.csv", index=False)

class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.linear = nn.Linear(4, 1)  # 4个特征输入到线性层

    def forward(self, x):
        return self.linear(x)
# 损失函数: Hinge Loss
criterion = nn.HingeEmbeddingLoss()

# 优化器: 梯度下降法
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
from sklearn.model_selection import train_test_split
import numpy as np

# 加载处理后的数据集
normalized_data = pd.read_csv("normalized_perfume_data.csv")

# 划分特征和标签
X = normalized_data.iloc[:, :-1].values  # 特征
y = normalized_data.iloc[:, -1].values   # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 在测试集上评估模型性能
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = torch.sign(outputs).detach().numpy()
    accuracy = np.mean(predicted == y_test)
    print(f'Accuracy on test set: {accuracy:.2f}')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 可视化决策边界
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制数据点
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

# 生成决策边界网格
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
x3_range = np.linspace(X[:, 2].min(), X[:, 2].max(), 10)
x1_mesh, x2_mesh, x3_mesh = np.meshgrid(x1_range, x2_range, x3_range)
X_mesh = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel(), x3_mesh.ravel()))
decision_boundary = model(torch.tensor(X_mesh, dtype=torch.float32)).detach().numpy().reshape(x1_mesh.shape)

# 绘制决策边界
ax.contour3D(x1_mesh, x2_mesh, x3_mesh, decision_boundary, 0, cmap='viridis')

# 设置图例
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('Decision Boundary')

plt.show()
