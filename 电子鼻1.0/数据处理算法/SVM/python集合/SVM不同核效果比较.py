import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

# 读取数据
train_data = pd.read_csv("train_dataclass.csv")
test_data = pd.read_csv("test_dataclass.csv")

# 提取特征和标签
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# 定义不同的SVM核函数
kernels = ['linear', 'poly', 'rbf']
kernel_names = ['Linear', 'Polynomial', 'RBF']

# 存储结果的列表
accuracies = []
training_times = []

# 训练并评估模型
for kernel in kernels:
    start_time = time.time()

    # 创建SVM模型
    svm = SVC(kernel=kernel)

    # 训练模型
    svm.fit(X_train, y_train)

    # 预测
    y_pred = svm.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # 计算训练时间
    training_time = time.time() - start_time
    training_times.append(training_time)

    print(f"Kernel: {kernel_names[kernels.index(kernel)]}")
    print(f"Accuracy: {accuracy}")
    print(f"Training Time: {training_time} seconds")
    print("--------------------------------------")

# 绘制效果比较区域图
plt.figure(figsize=(10, 6))
plt.bar(kernel_names, accuracies, color='skyblue')
plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.title('Comparison of SVM Kernels')
plt.ylim(0, 1)
plt.show()
