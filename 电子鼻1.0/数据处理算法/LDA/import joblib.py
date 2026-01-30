import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from joblib import dump


# 加载八个Excel表格数据
excel_files = [
    r"C:\Users\123\Desktop\Creed.xlsx",
    r"C:\Users\123\Desktop\MY WAY.xlsx",
    r"C:\Users\123\Desktop\LIBRE.xlsx",
    r"C:\Users\123\Desktop\ROSE OF NO MAN’S LAND.xlsx",
    r"C:\Users\123\Desktop\WHITE TEA.xlsx",
    r"C:\Users\123\Desktop\反转巴黎.xlsx",
    r"C:\Users\123\Desktop\英.xlsx",
    r"C:\Users\123\Desktop\牧.xlsx"
]

# 初始化一个空列表来存储所有类别的数据
all_data = []

# 逐个加载每个 Excel 表格数据，并为每个类别添加一个标签列
for i, file in enumerate(excel_files):
    df = pd.read_excel(file)
    df['label'] = i  # 为每个类别添加一个标签列，值为类别索引
    all_data.append(df)

# 将所有类别的数据合并为一个DataFrame
data = pd.concat(all_data, ignore_index=True)

# 特征列
X = data.iloc[:, :-1]  # 所有列除了最后一列（标签列）都是特征

# 标签列
y = data['label']  # 使用 'label' 列作为标签

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 PCA 将数据降至二维空间
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
# 使用相同的 PCA 变换将测试数据降至二维
X_test_pca = pca.transform(X_test)


# 初始化并训练 LDA 模型
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_pca, y_train)

# 使用训练好的 LDA 模型对降维后的数据进行投影
X_train_lda = lda.transform(X_train_pca)

# 绘制散点图
plt.figure(figsize=(10, 6))
for label in np.unique(y_train):
    plt.scatter(X_train_lda[y_train == label, 0], X_train_lda[y_train == label, 1], label=label)

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA Projection of PCA-reduced Data')
plt.legend(title='Class')
plt.grid(True)
plt.show()

# 定义特征名称
feature_names = ['NO2', 'C2H5CH', 'VOC', 'CO']

# 创建 LDA 模型
lda = LinearDiscriminantAnalysis(store_covariance=True)

# 使用特征名称和数据来训练模型
lda.fit(X_train, y_train)

# 在需要时引用特征名称
print("Feature names:", feature_names)


# 在测试集上进行预测
y_pred = lda.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print(f'准确度：{accuracy}')

# 输出分类报告
class_report = classification_report(y_test, y_pred, output_dict=True)
print('分类报告：')
print(class_report)
# 提取准确率、召回率和 F1 值
precision = [class_report[label]['precision'] for label in class_report.keys() if label != 'accuracy']
recall = [class_report[label]['recall'] for label in class_report.keys() if label != 'accuracy']
f1_score = [class_report[label]['f1-score'] for label in class_report.keys() if label != 'accuracy']
labels = [label for label in class_report.keys() if label != 'accuracy']

# 可视化
plt.figure(figsize=(10, 5))
plt.bar(labels, precision, color='blue', alpha=0.5, label='Precision')
plt.bar(labels, recall, color='green', alpha=0.5, label='Recall')
plt.bar(labels, f1_score, color='red', alpha=0.5, label='F1 Score')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Classification Report')
plt.legend()
plt.show()

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print('混淆矩阵：')
print(conf_matrix)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 定义绘制混淆矩阵的函数
def plot_confusion_matrix(conf_matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()


# 绘制混淆矩阵
plot_confusion_matrix(conf_matrix, excel_files)
