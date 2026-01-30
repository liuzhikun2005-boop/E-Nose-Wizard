
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np

# 忽略特定警告
import warnings

warnings.filterwarnings('ignore', message="X does not have valid feature names, but SVC was fitted with feature names")

# 加载保存的模型
best_svm_model = joblib.load('svm_model.pkl')

def classify_input():
    # 获取用户输入
    user_input = entry.get()

    # 将输入字符串转换为numpy数组
    input_values = np.array([float(x) for x in user_input.split()]).reshape(1, -1)

    # 使用加载的模型进行预测
    prediction = best_svm_model.predict(input_values)

    # 弹出消息框显示预测的类别
    messagebox.showinfo("预测结果", f'预测的类别: {prediction[0]}')

# 创建主窗口
root = tk.Tk()
root.title("气味分类器")

# 设置窗口大小和位置
root.geometry("400x200+200+200")

# 创建ttk样式对象
style = ttk.Style()

# 设置渐变色背景
style.configure('TFrame', background='#003366')  # 设置渐变色背景，这是蓝色的渐变

# 设置ttk样式外观
style.theme_use('clam')

# 添加标题标签
title_label = ttk.Label(root, text="欢迎使用气味分类器", font=('Arial', 18), foreground="white", background='#003366')
title_label.pack(pady=10)

# 添加标签和输入框
label = ttk.Label(root, text="请输入4个特征值，用空格分隔:", font=('Arial', 12), foreground="white", background='#003366')
label.pack()
entry = ttk.Entry(root, font=('Arial', 12))
entry.pack(pady=5)

# 添加按钮
button = ttk.Button(root, text="进行分类", command=classify_input)
button.pack(pady=10)

# 运行主事件循环
root.mainloop()

