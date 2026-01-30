import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#线性归一化
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 加载数据
# 加载数据，跳过第一行
data = pd.read_csv("dataclass.csv", skiprows=1)

features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# 数据归一化
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# 保存新的归一化数据集
normalized_data = pd.DataFrame(features_scaled)
normalized_data['label'] = labels
normalized_data.to_csv('normalized_perfume_data.csv', index=False, header=False)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# 训练SVM模型
model = svm.SVC(kernel='linear', C=1.0)  # 使用线性核
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# 绘制决策边界 - 仅限于两个特征可视化
def plot_decision_boundaries(X, Y, model_class, **model_params):
    """
    Function to plot the decision boundaries of a classification model.
    """
    reducer = PCA(n_components=2)
    X_reduced = reducer.fit_transform(X)

    model = model_class(**model_params)
    model.fit(X_reduced, Y)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y, s=50, cmap='autumn')

    ax = plt.gca()#获取当前的子图，要有网格才能画出决策边界
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 30),#最小值最大值，三十个数
                         np.linspace(ylim[0], ylim[1], 30))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
#z是距离
    # plot decision boundaries
    plt.contour(xx, yy, Z, colors='b', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    plt.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


from sklearn.decomposition import PCA

plot_decision_boundaries(features_scaled, labels, svm.SVC, kernel='linear', C=1.0)
