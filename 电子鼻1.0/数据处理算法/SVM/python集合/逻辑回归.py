import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# 读取训练集和测试集数据
train_data = pd.read_csv('train_dataclass.csv')
test_data = pd.read_csv('test_dataclass.csv')

# 提取特征和标签
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# 归一化处理数据
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)


# 将标签转换为独热编码
def one_hot_encode(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    encoded_labels = np.zeros((num_labels, num_classes))
    encoded_labels.flat[index_offset + labels.ravel()] = 1
    return encoded_labels


num_classes = len(np.unique(y_train))
y_train_encoded = one_hot_encode(y_train, num_classes)


# 定义逻辑回归模型
class LogisticRegression:
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.zeros((1, output_dim))
        self.learning_rate = learning_rate

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_pred.shape[0]
        loss = -1 / m * np.sum(y_true * np.log(y_pred))
        return loss

    def train(self, X, y_true, epochs):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            z = np.dot(X, self.W) + self.b
            y_pred = self.softmax(z)

            # Compute loss
            loss = self.cross_entropy_loss(y_pred, y_true)
            losses.append(loss)

            # Backpropagation
            dz = y_pred - y_true
            dw = np.dot(X.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)

            # Update parameters
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

        return losses

    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        y_pred = self.softmax(z)
        return np.argmax(y_pred, axis=1)


# 设置超参数
input_dim = X_train_normalized.shape[1]
output_dim = num_classes
learning_rate = 0.0001
epochs =5000

# 创建并训练模型
model = LogisticRegression(input_dim, output_dim, learning_rate)
losses = model.train(X_train_normalized, y_train_encoded, epochs)

# 测试模型

y_pred = model.predict(X_test_normalized)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# 绘制准确度随训练周期变化的折线图
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.show()
