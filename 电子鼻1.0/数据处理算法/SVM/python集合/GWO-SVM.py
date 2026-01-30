
# 读取数据集
#0-香水一MYWAY
#1-可乐
#2-咖啡
#3-酸奶
#4-啤酒酒精
#5-水蜜桃汽水
#6-冰红茶
#7-室友香水
#8-libre
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: 加载数据集
dataclass = pd.read_csv('dataclass.csv')

# Step 2: 归一化处理
def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

dataclass_normalized = dataclass.apply(normalize_data)

# 为了方便，假设最后一列是标签，前面的列是特征
features = dataclass_normalized.iloc[:, :-1].values
labels = dataclass_normalized.iloc[:, -1].values

# 转换为 PyTorch 张量
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# Step 3: 定义支持向量机 (SVM) 模型
class SVM(torch.nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.linear = torch.nn.Linear(4, 1)  # 假设有4个特征

    def forward(self, x):
        return self.linear(x)

# Step 4: 定义灰狼优化算法 (GWO)
class GWO:
    def __init__(self, model, features, labels, num_iter, alpha=0.1):
        self.model = model
        self.features = features
        self.labels = labels
        self.num_iter = num_iter
        self.alpha = alpha

    def optimize(self):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        losses = []
        accuracies = []

        for _ in range(self.num_iter):
            optimizer.zero_grad()
            outputs = self.model(self.features)
            loss = criterion(outputs, self.labels.view(-1, 1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # 计算准确度（这里是简单的计算预测值和真实值相等的比例）
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == self.labels.view(-1, 1)).float().mean().item()
            accuracies.append(accuracy)

        return losses, accuracies

# Step 5: 运行 GWO 优化算法并记录损失和准确度
model = SVM()
gwo_optimizer = GWO(model, features, labels, num_iter=100)
losses, accuracies = gwo_optimizer.optimize()
print("模型精确度:", accuracies)
# Step 6: 绘制损失和准确度随着迭代次数的变化曲线
plt.figure(figsize=(12, 6))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss', color='blue')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.legend()
plt.grid(True)

# 绘制准确度曲线
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy', color='red')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy over Iterations')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
