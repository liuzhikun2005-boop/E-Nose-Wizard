import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 读取测试数据集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv("test.csv")

# 划分训练集和测试集
X_train = train_data.iloc[:, :-1]  # 训练集特征
y_train = train_data.iloc[:, -1]   # 训练集标签
X_test = test_data.iloc[:, :-1]  # 测试集特征
y_test = test_data.iloc[:, -1]   # 测试集标签

# 定义随机森林模型
rf_classifier = RandomForestClassifier(random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2']
}

# 使用网格搜索交叉验证方法寻找最优参数组合
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最优参数组合
print("Best Parameters:", grid_search.best_params_)

# 使用最优参数组合的模型进行分类
best_rf_classifier = grid_search.best_estimator_

# 进行交叉验证并计算性能指标
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'roc_auc_ovr_weighted']
cv_results = cross_validate(best_rf_classifier, X_train, y_train, scoring=scoring, cv=5)

# 输出交叉验证结果
print("\nCross Validation Results:")
for metric in scoring:
    print(f"{metric}: {cv_results['test_'+metric].mean()} (±{cv_results['test_'+metric].std()})")

# 在测试集上进行预测
best_rf_classifier.fit(X_train, y_train)
y_pred = best_rf_classifier.predict(X_test)

# 计算模型性能指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("\nTest Set Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# 可视化ROC曲线和计算AUC
y_pred_prob = best_rf_classifier.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
