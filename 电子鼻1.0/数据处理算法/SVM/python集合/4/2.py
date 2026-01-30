import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# 加载保存的模型
loaded_model = joblib.load('svm_model.pkl')

# 加载测试数据
test_data = pd.read_csv('test_dataclass.csv')

# 提取特征和标签
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

# 使用加载的模型进行预测
predictions = loaded_model.predict(X_test)

# 计算测试集上的准确率
accuracy = accuracy_score(y_test, predictions)
print(f'测试集上的准确率: {accuracy}')

