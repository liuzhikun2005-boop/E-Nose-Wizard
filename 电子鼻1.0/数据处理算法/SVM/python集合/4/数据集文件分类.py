import pandas as pd
import joblib

# 加载保存的模型
best_svm_model = joblib.load('svm_model.pkl')

# 加载新的输入数据
new_input_data = pd.read_csv('input_data.csv')

# 使用加载的模型进行预测
predictions = best_svm_model.predict(new_input_data)

# 将预测结果保存到文件
output = pd.DataFrame({'Predicted': predictions})
output.to_csv('predicted_results.csv', index=False)

print('预测结果已保存到 predicted_results.csv')
