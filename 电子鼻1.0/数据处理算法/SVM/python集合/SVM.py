from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# 读取数据集
data = pd.read_csv("data.csv")  # 替换为你的数据集路径

# 提取特征和标签
X = data.iloc[:, :-1]  # 前四列是特征
y = data.iloc[:, -1]   # 最后一列是标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# 定义SVM模型,高维可用高斯，建议细看模型参数的原理
svm = SVC(kernel='poly')  # 这里使用多项式核，你也可以尝试其他核函数和参数，默认参数degree=3，gamma=1/4,c=1.0

# 训练模型
svm.fit(X_train, y_train)

# 预测测试集
y_pred = svm.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print("模型准确度:", accuracy)
