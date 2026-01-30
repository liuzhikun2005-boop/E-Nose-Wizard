
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
import pandas as pd

# 读取训练集和测试集数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 提取特征和标签
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# 归一化处理数据
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# PCA降维至二维
pca = PCA(n_components=2)  # PCA降至二维
X_train_pca = pca.fit_transform(X_train_normalized)
X_test_pca = pca.transform(X_test_normalized)

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
        loss = -1 / m * np.sum(y_true * np.log(y_pred + 1e-9)) # Adding a small epsilon to prevent division by zero
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
            dw = 1 / X.shape[0] * np.dot(X.T, dz)
            db = 1 / X.shape[0] * np.sum(dz, axis=0, keepdims=True)

            # Update parameters
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

        return losses

# Initialize and train the logistic regression model
input_dim = X_train_pca.shape[1]
output_dim = num_classes
lr_model = LogisticRegression(input_dim, output_dim)
epochs = 200
losses = lr_model.train(X_train_pca, y_train_encoded, epochs)

# Plot the loss curve
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()
