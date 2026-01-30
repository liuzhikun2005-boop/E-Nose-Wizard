import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 加载训练数据和测试数据
train_data = pd.read_csv('train_dataclass.csv')
test_data = pd.read_csv('test_dataclass.csv')

# 提取特征和标签
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# 使用GridSearchCV进行网格搜索
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

# 输出最佳参数
print(f'最佳参数: {grid.best_params_}')

# 输出每种模型的准确度
cv_results = pd.DataFrame(grid.cv_results_)
print(cv_results[['param_C', 'param_gamma', 'param_kernel', 'mean_test_score']])

# 可视化准确度
accuracies_df = cv_results[['param_C', 'param_gamma', 'param_kernel', 'mean_test_score']]
accuracies_df['C'] = accuracies_df['param_C']
accuracies_df['Gamma'] = accuracies_df['param_gamma']
accuracies_df['Kernel'] = accuracies_df['param_kernel']

pivot_table = accuracies_df.pivot_table(index='C', columns='Gamma', values='mean_test_score')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu")
plt.title('Accuracy Heatmap')
plt.xlabel('Gamma')
plt.ylabel('C')
plt.show()

# 输出测试集上的准确率
best_model = grid.best_estimator_
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'测试集上的准确率: {accuracy}')

# 输出预测结果
result = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(result)
