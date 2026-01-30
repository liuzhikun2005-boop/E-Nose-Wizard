import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
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

# PCA降维
pca = PCA(n_components=3)
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
            dw = np.dot(X.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)

            # Update parameters
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return losses

# 训练模型
input_dim = X_train_pca.shape[1]
output_dim = num_classes
model = LogisticRegression(input_dim, output_dim)
losses = model.train(X_train_pca, y_train_encoded, epochs=1000)

# 预测并计算准确率
def predict(X, model):
    z = np.dot(X, model.W) + model.b
    y_pred = model.softmax(z)
    return np.argmax(y_pred, axis=1)

y_pred_train = predict(X_train_pca, model)
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Training Accuracy:", accuracy_train)

y_pred_test = predict(X_test_pca, model)
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Accuracy:", accuracy_test)

# 生成ROC曲线
def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

y_score_train = model.softmax(np.dot(X_train_pca, model.W) + model.b)[:, 1]
plot_roc_curve(y_train, y_score_train)

y_score_test = model.softmax(np.dot(X_test_pca, model.W) + model.b)[:, 1]
plot_roc_curve(y_test, y_score_test)

# 可视化降维后的数据
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Add more colors if needed
markers = ['o', '^', 's', 'x', 'D', 'P', '*']  # Add more markers if needed
for i in range(num_classes):
    ax.scatter(X_train_pca[y_train==i, 0], X_train_pca[y_train==i, 1], X_train_pca[y_train==i, 2],
               c=colors[i % len(colors)], marker=markers[i % len(markers)], label=str(i))
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA Visualization')
plt.legend()
plt.show()
