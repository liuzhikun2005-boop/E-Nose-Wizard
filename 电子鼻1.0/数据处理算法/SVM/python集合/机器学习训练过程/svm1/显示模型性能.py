import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Create a Tkinter window
root = tk.Tk()

title_label = tk.Label(root, text="SVM1算法模型性能评估如下：", font=("Arial", 14, "bold"))
root.title("SVM算法模型性能评估如下")

# 加载保存的模型
loaded_model = joblib.load('svm_model.pkl')

# 加载测试数据
test_data = pd.read_csv('test_dataclass.csv')

# 提取特征和标签
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

# 使用加载的模型进行预测
predictions = loaded_model.predict(X_test)

# 计算测试集上的准确率、召回率、F1值和混淆矩阵
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
conf_matrix = confusion_matrix(y_test, predictions)

# Display results in Tkinter window
results_label = tk.Label(root, text="Test Set Evaluation Results", font=("Helvetica", 16))
results_label.pack()

accuracy_label = tk.Label(root, text=f'Accuracy: {accuracy:.4f}')
accuracy_label.pack()

recall_label = tk.Label(root, text=f'Recall: {recall:.4f}')
recall_label.pack()

f1_label = tk.Label(root, text=f'F1 Score: {f1:.4f}')
f1_label.pack()

conf_matrix_label = tk.Label(root, text='Confusion Matrix:')
conf_matrix_label.pack()

conf_matrix_text = tk.Text(root)
conf_matrix_text.insert(tk.END, str(conf_matrix))
conf_matrix_text.pack()

# 加载训练数据
train_data = pd.read_csv('train_dataclass.csv')

# 提取特征和标签
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 使用加载的模型进行验证集上的预测
val_predictions = loaded_model.predict(X_val)

# 计算验证集上的准确率、召回率、F1 值和混淆矩阵
val_accuracy = accuracy_score(y_val, val_predictions)
val_recall = recall_score(y_val, val_predictions, average='weighted')
val_f1 = f1_score(y_val, val_predictions, average='weighted')
val_conf_matrix = confusion_matrix(y_val, val_predictions)

# Display validation set results in Tkinter window
val_results_label = tk.Label(root, text="Validation Set Evaluation Results", font=("Helvetica", 16))
val_results_label.pack()

val_accuracy_label = tk.Label(root, text=f'Accuracy: {val_accuracy:.4f}')
val_accuracy_label.pack()

val_recall_label = tk.Label(root, text=f'Recall: {val_recall:.4f}')
val_recall_label.pack()

val_f1_label = tk.Label(root, text=f'F1 Score: {val_f1:.4f}')
val_f1_label.pack()

val_conf_matrix_label = tk.Label(root, text='Confusion Matrix:')
val_conf_matrix_label.pack()

val_conf_matrix_text = tk.Text(root)
val_conf_matrix_text.insert(tk.END, str(val_conf_matrix))
val_conf_matrix_text.pack()

root.mainloop()
