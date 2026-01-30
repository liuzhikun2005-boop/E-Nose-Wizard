import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 加载训练数据和测试数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 提取特征和标签
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

# 使用指定参数创建SVM分类器
svm_model = SVC(C= 0.1, gamma=1, kernel='linear')

# 训练模型
svm_model.fit(X_train, y_train)

# 使用模型进行预测
predictions = svm_model.predict(X_test)

# 计算测试集上的准确率
accuracy = accuracy_score(y_test, predictions)
print(f'测试集上的准确率: {accuracy}')

# 计算并输出混淆矩阵
conf_matrix = confusion_matrix(y_test, predictions)
print('混淆矩阵:')
print(conf_matrix)

# 用图表示混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
#plt.savefig('confusion_matrix.png')
plt.show()

# 保存模型
joblib.dump(svm_model, 'svm_model11.pkl')

# 输出预测结果
result = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(result)
