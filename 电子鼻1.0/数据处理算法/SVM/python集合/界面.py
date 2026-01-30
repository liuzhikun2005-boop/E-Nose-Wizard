import tkinter as tk
from tkinter import messagebox
import KNN类别训练 # 导入knn文件中的函数
import RF  # 导入随机森林文件中的函数
import SVM # 导入svm文件中的函数

def knn_button_click():
    result = KNN类别训练()  # 调用knn文件中的函数
    output_text.set(result)  # 将结果输出到界面上

def random_forest_button_click():
    result = RF()  # 调用随机森林文件中的函数
    output_text.set(result)  # 将结果输出到界面上

def svm_button_click():
    # 跳转到下一个界面或执行其他操作
    messagebox.showinfo("提示", "将跳转到下一个界面")

# 创建主界面
root = tk.Tk()
root.title("Machine Learning Algorithms")

# 创建按钮
knn_button = tk.Button(root, text="KNN", command=knn_button_click)
knn_button.pack()

random_forest_button = tk.Button(root, text="Random Forest", command=random_forest_button_click)
random_forest_button.pack()

svm_button = tk.Button(root, text="SVM", command=svm_button_click)
svm_button.pack()

# 创建用于输出结果的文本框
output_text = tk.StringVar()
output_label = tk.Label(root, textvariable=output_text)
output_label.pack()

root.mainloop()
