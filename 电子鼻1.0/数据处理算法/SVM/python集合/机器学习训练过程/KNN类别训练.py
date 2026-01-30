import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 读取训练集和测试集数据
train_data = pd.read_csv('train_dataclass.csv')
test_data = pd.read_csv('test_dataclass.csv')

# 提取特征和标签
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# 数据归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# 训练KNN分类器
k = 6
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# 在验证集上进行预测
y_val_pred = knn_classifier.predict(X_val)

# 在测试集上进行预测
y_test_pred = knn_classifier.predict(X_test_scaled)

# 计算在验证集上的性能指标
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')
val_f1 = f1_score(y_val, y_val_pred, average='weighted')
val_conf_matrix = confusion_matrix(y_val, y_val_pred)

# 计算在测试集上的性能指标
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

# 创建Tkinter窗口
root = tk.Tk()
root.title("KNN Classifier Performance Metrics")


# 标题
title_label = tk.Label(root, text="KNN算法模型性能评估如下：", font=("Arial", 14, "bold"))
title_label.pack(padx=10, pady=10)

# 显示性能指标
metrics_frame = ttk.Frame(root)
metrics_frame.pack(padx=10, pady=10)

# 创建性能指标标签
tk.Label(metrics_frame, text="性能指标", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=5, pady=5)
tk.Label(metrics_frame, text="验证集", font=("Arial", 12, "bold")).grid(row=0, column=1, padx=5, pady=5)
tk.Label(metrics_frame, text="测试集", font=("Arial", 12, "bold")).grid(row=0, column=2, padx=5, pady=5)

# Accuracy
tk.Label(metrics_frame, text="Accuracy", font=("Arial", 10)).grid(row=1, column=0, padx=5, pady=5)
tk.Label(metrics_frame, text="{:.2f}".format(val_accuracy), font=("Arial", 10)).grid(row=1, column=1, padx=5, pady=5)
tk.Label(metrics_frame, text="{:.2f}".format(test_accuracy), font=("Arial", 10)).grid(row=1, column=2, padx=5, pady=5)

# Precision
tk.Label(metrics_frame, text="Precision", font=("Arial", 10)).grid(row=2, column=0, padx=5, pady=5)
tk.Label(metrics_frame, text="{:.2f}".format(val_precision), font=("Arial", 10)).grid(row=2, column=1, padx=5, pady=5)
tk.Label(metrics_frame, text="{:.2f}".format(test_precision), font=("Arial", 10)).grid(row=2, column=2, padx=5, pady=5)

# Recall
tk.Label(metrics_frame, text="Recall", font=("Arial", 10)).grid(row=3, column=0, padx=5, pady=5)
tk.Label(metrics_frame, text="{:.2f}".format(val_recall), font=("Arial", 10)).grid(row=3, column=1, padx=5, pady=5)
tk.Label(metrics_frame, text="{:.2f}".format(test_recall), font=("Arial", 10)).grid(row=3, column=2, padx=5, pady=5)

# F1 Score
tk.Label(metrics_frame, text="F1 Score", font=("Arial", 10)).grid(row=4, column=0, padx=5, pady=5)
tk.Label(metrics_frame, text="{:.2f}".format(val_f1), font=("Arial", 10)).grid(row=4, column=1, padx=5, pady=5)
tk.Label(metrics_frame, text="{:.2f}".format(test_f1), font=("Arial", 10)).grid(row=4, column=2, padx=5, pady=5)

# Confusion Matrix
# You can choose to display the confusion matrix if needed
# 在窗口中显示混淆矩阵
conf_matrix_frame = ttk.Frame(root)
conf_matrix_frame.pack(padx=10, pady=10)

conf_matrix_label = tk.Label(conf_matrix_frame, text="混淆矩阵", font=("Arial", 12, "bold"))
conf_matrix_label.pack(pady=5)

# 创建Canvas用于绘制混淆矩阵
conf_matrix_canvas = tk.Canvas(conf_matrix_frame, width=300, height=300, bg="white")
conf_matrix_canvas.pack()

# 绘制混淆矩阵
matrix_size = len(val_conf_matrix)
cell_width = 300 / matrix_size
cell_height = 300 / matrix_size

for i in range(matrix_size):
    for j in range(matrix_size):
        x0 = i * cell_width
        y0 = j * cell_height
        x1 = x0 + cell_width
        y1 = y0 + cell_height
        conf_matrix_canvas.create_rectangle(x0, y0, x1, y1, outline="black", fill="white")
        conf_matrix_canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(val_conf_matrix[i][j]))

root.mainloop()