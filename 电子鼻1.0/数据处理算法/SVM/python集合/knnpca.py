import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
#from sklearn.metrics import plot_roc_curve

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

# 初始化列表以存储不同k值下的准确率（未进行归一化）
accuracies_unscaled = []

# 初始化混淆矩阵
conf_matrices = []
conf_matrices_unscaled = []

# 初始化ROC曲线数据
roc_data = []
roc_data_unscaled = []

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

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrices.append(conf_matrix)

    # 计算ROC曲线数据
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    y_pred_proba = knn.predict_proba(X_test_scaled)
    fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_pred_proba.ravel())
    roc_auc = roc_auc_score(y_true_binarized, y_pred_proba, average='micro')
    roc_data.append((fpr, tpr, roc_auc))

    # 训练KNN分类器（未进行归一化）
    knn_unscaled = KNeighborsClassifier(n_neighbors=k)
    knn_unscaled.fit(X_train, y_train)

    # 预测验证集数据（未进行归一化）
    y_pred_unscaled = knn_unscaled.predict(X_test)

    # 计算准确率（未进行归一化）
    accuracy_unscaled = accuracy_score(y_true, y_pred_unscaled)
    accuracies_unscaled.append(accuracy_unscaled)

    # 计算混淆矩阵（未进行归一化）
    conf_matrix_unscaled = confusion_matrix(y_true, y_pred_unscaled)
    conf_matrices_unscaled.append(conf_matrix_unscaled)

    # 计算ROC曲线数据（未进行归一化）
    y_pred_proba_unscaled = knn_unscaled.predict_proba(X_test)
    fpr_unscaled, tpr_unscaled, _ = roc_curve(y_true_binarized.ravel(), y_pred_proba_unscaled.ravel())
    roc_auc_unscaled = roc_auc_score(y_true_binarized, y_pred_proba_unscaled, average='micro')
    roc_data_unscaled.append((fpr_unscaled, tpr_unscaled, roc_auc_unscaled))


# 可视化ROC曲线
plt.subplot(1, 2, 2)
for i, k in enumerate(k_values):
    fpr, tpr, roc_auc = roc_data[i]
    fpr_unscaled, tpr_unscaled, roc_auc_unscaled = roc_data_unscaled[i]
    plt.plot(fpr, tpr, label=f'With Scaling (K={k}, AUC={roc_auc:.2f})')
    plt.plot(fpr_unscaled, tpr_unscaled, linestyle='--', label=f'Without Scaling (K={k}, AUC={roc_auc_unscaled:.2f})')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.legend()
plt.show()


plt.show()
