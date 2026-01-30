import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 创建Tkinter窗口
window = tk.Tk()
window.title("Classifier Evaluation")
window.geometry("800x600")

# 读取数据集
train_data = pd.read_csv('train_dataclass.csv')
test_data = pd.read_csv('test_dataclass.csv')

# 数据预处理
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# 归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA降维
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# 定义模型
class SVM2(nn.Module):
    def __init__(self):
        super(SVM2, self).__init__()
        self.linear = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear(x)


# 训练KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train_pca, y_train)

# 训练逻辑回归
lr = LogisticRegression(max_iter=5000, solver='lbfgs', multi_class='multinomial')
lr.fit(X_train_pca, y_train)

# 训练SVM1
svm1 = SVC(C=0.1, gamma=1, kernel='linear')
svm1.fit(X_train_pca, y_train)

# 训练SVM2
svm2 = SVM2()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(svm2.parameters(), lr=0.01)

X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)


# KNN按钮点击事件处理函数
def knn_clicked():
    y_pred = knn.predict(X_test_pca)
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    acc_test = accuracy_score(y_test, y_pred)
    acc_val = knn.score(X_train_pca, y_train)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    # 在窗口上显示结果
    result_text = f"Confusion Matrix:\n{cm}\n\nRecall: {recall}\n\nTest Accuracy: {acc_test}\n\nValidation Accuracy: {acc_val}\n\nAUC: {auc}"
    result_label.config(text=result_text)


# 逻辑回归按钮点击事件处理函数
def lr_clicked():
    y_pred = lr.predict(X_test_pca)
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    acc_test = accuracy_score(y_test, y_pred)
    acc_val = lr.score(X_train_pca, y_train)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    # 在窗口上显示结果
    result_text = f"Confusion Matrix:\n{cm}\n\nRecall: {recall}\n\nTest Accuracy: {acc_test}\n\nValidation Accuracy: {acc_val}\n\nAUC: {auc}"
    result_label.config(text=result_text)


# SVM1按钮点击事件处理函数
def svm1_clicked():
    y_pred = svm1.predict(X_test_pca)
    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    acc_test = accuracy_score(y_test, y_pred)
    acc_val = svm1.score(X_train_pca, y_train)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    # 在窗口上显示结果
    result_text = f"Confusion Matrix:\n{cm}\n\nRecall: {recall}\n\nTest Accuracy: {acc_test}\n\nValidation Accuracy: {acc_val}\n\nAUC: {auc}"
    result_label.config(text=result_text)


# SVM2按钮点击事件处理函数
def svm2_clicked():
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = svm2(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = torch.round(torch.sigmoid(svm2(torch.tensor(X_test_pca, dtype=torch.float32))))
        y_pred = y_pred.numpy().astype(int)

    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    acc_test = accuracy_score(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    # 在窗口上显示结果
    result_text = f"Confusion Matrix:\n{cm}\n\nRecall: {recall}\n\nTest Accuracy: {acc_test}\n\nAUC: {auc}"
    result_label.config(text=result_text)


# 创建按钮
knn_button = tk.Button(window, text="KNN", command=knn_clicked)
knn_button.pack()

lr_button = tk.Button(window, text="Logistic Regression", command=lr_clicked)
lr_button.pack()

svm1_button = tk.Button(window, text="SVM1", command=svm1_clicked)
svm1_button.pack()

svm2_button = tk.Button(window, text="SVM2", command=svm2_clicked)
svm2_button.pack()

def plot_roc_curve(y_true, y_score, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - ' + model_name)
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - ' + model_name)
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted Negative', 'Predicted Positive'])
    plt.yticks([0, 1], ['Actual Negative', 'Actual Positive'])

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def calculate_recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    return recall

def calculate_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def evaluate_model(model, X_test, y_test, model_name):
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            y_score = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    else:
        y_score = model.predict_proba(X_test)[:, 1]

    y_pred = model.predict(X_test)
    plot_roc_curve(y_test, y_score, model_name)
    plot_confusion_matrix(y_test, y_pred, model_name)
    recall = calculate_recall(y_test, y_pred)
    accuracy = calculate_accuracy(y_test, y_pred)
    return recall, accuracy

def plot_results(recall, accuracy):
    result_label.config(text=f"Recall: {recall}\nAccuracy: {accuracy}")

def evaluate_and_plot(model_name):
    if model_name == "KNN":
        recall, accuracy = evaluate_model(knn, X_test_pca, y_test, "KNN")
    elif model_name == "Logistic Regression":
        recall, accuracy = evaluate_model(lr, X_test_pca, y_test, "Logistic Regression")
    elif model_name == "SVM1":
        recall, accuracy = evaluate_model(svm1, X_test_pca, y_test, "SVM1")
    elif model_name == "SVM2":
        recall, accuracy = evaluate_model(svm2, X_test_pca, y_test, "SVM2")
    plot_results(recall, accuracy)

# 创建按钮
knn_button = tk.Button(window, text="KNN", command=lambda: evaluate_and_plot("KNN"))
knn_button.pack()

lr_button = tk.Button(window, text="Logistic Regression", command=lambda: evaluate_and_plot("Logistic Regression"))
lr_button.pack()

svm1_button = tk.Button(window, text="SVM1", command=lambda: evaluate_and_plot("SVM1"))
svm1_button.pack()

svm2_button = tk.Button(window, text="SVM2", command=lambda: evaluate_and_plot("SVM2"))
svm2_button.pack()

# 创建用于显示结果的标签
result_label = tk.Label(window, text="")
result_label.pack()

window.mainloop()
