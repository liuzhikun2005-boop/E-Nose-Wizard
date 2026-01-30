import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def display_metrics(metrics_text, conf_matrix):
    root = tk.Tk()
    root.title("Logistic Regression Performance Metrics")

    # 标题
    title_label = tk.Label(root, text="逻辑回归算法模型性能评估如下：", font=("Arial", 14, "bold"))
    title_label.pack(padx=10, pady=10)

    # 性能指标表格
    metrics_frame = ttk.Frame(root)
    metrics_frame.pack(padx=10, pady=5)

    metrics_tree = ttk.Treeview(metrics_frame)
    metrics_tree["columns"] = ("Metric", "Validation Set", "Test Set")
    metrics_tree.heading("#0", text="", anchor="center")
    metrics_tree.heading("Metric", text="Metric", anchor="center")
    metrics_tree.heading("Validation Set", text="Validation Set", anchor="center")
    metrics_tree.heading("Test Set", text="Test Set", anchor="center")

    for metric, val, test in metrics_text:
        metrics_tree.insert("", "end", text="", values=(metric, val, test))

    metrics_tree.pack(fill="both", expand=True)

    # 混淆矩阵
    conf_matrix_frame = ttk.Frame(root)
    conf_matrix_frame.pack(padx=10, pady=5)

    conf_matrix_label = tk.Label(conf_matrix_frame, text="Confusion Matrix", font=("Arial", 14, "bold"))
    conf_matrix_label.pack(pady=5)

    # Create a canvas to draw the confusion matrix
    canvas = tk.Canvas(conf_matrix_frame, width=300, height=300)
    canvas.pack()

    # Calculate cell width and height
    cell_width = 300 / len(conf_matrix)
    cell_height = 300 / len(conf_matrix)

    # Draw the confusion matrix
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            x0 = j * cell_width
            y0 = i * cell_height
            x1 = x0 + cell_width
            y1 = y0 + cell_height
            canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black")
            canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(conf_matrix[i][j]))

    root.mainloop()

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

# 定义逻辑回归模型
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# 构建管道
pipeline = make_pipeline(StandardScaler(), model)

# 训练模型
pipeline.fit(X_train, y_train)

# 在验证集上进行预测
y_val_pred = pipeline.predict(X_val)

# 在测试集上进行预测
y_test_pred = pipeline.predict(X_test_scaled)

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

# 构建性能指标文本
metrics_text = [
    ("Accuracy", val_accuracy, test_accuracy),
    ("Precision", val_precision, test_precision),
    ("Recall", val_recall, test_recall),
    ("F1 Score", val_f1, test_f1)
]

# 显示性能指标和混淆矩阵
display_metrics(metrics_text, test_conf_matrix)
