import tkinter as tk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class YourClass:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Model Evaluation")

        # Button to evaluate KNN model
        self.knn_button = tk.Button(self.window, text="KNN Model Evaluation", command=self.evaluate_knn)
        self.knn_button.pack()

        # Button to evaluate SVM model
        self.svm_button = tk.Button(self.window, text="SVM Model Evaluation", command=self.evaluate_svm)
        self.svm_button.pack()

        # Button to evaluate Logistic Regression model
        self.logreg_button = tk.Button(self.window, text="Logistic Regression Evaluation",
                                       command=self.evaluate_logreg)
        self.logreg_button.pack()

    def evaluate_knn(self):
        import tkinter as tk
        from tkinter import ttk
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler

        # 读取训练集和测试集数据
        train_data = pd.read_csv('train_dataclass.csv')
        test_data = pd.read_csv('test_dataclass.csv')

        # 提取特征和标签
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # 数据归一化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

        # 训练KNN分类器
        k = 6
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train, y_train)

        # 在验证集上进行预测
        y_val_pred = knn_classifier.predict(X_val)

        # 在测试集上进行预测
        y_test_pred = knn_classifier.predict(X_test_scaled)

        # 计算在验证集上的性能指标
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted')
        val_recall = recall_score(y_val, y_val_pred, average='weighted')
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        val_conf_matrix = confusion_matrix(y_val, y_val_pred)

        # 计算在测试集上的性能指标
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_conf_matrix = confusion_matrix(y_test, y_test_pred)

        # 创建Tkinter窗口
        root = tk.Tk()
        root.title("KNN Classifier Performance Metrics")

        # 标题
        title_label = tk.Label(root, text="KNN算法模型性能评估如下：", font=("Arial", 14, "bold"))
        title_label.pack(padx=10, pady=10)

        # 显示性能指标
        metrics_frame = ttk.Frame(root)
        metrics_frame.pack(padx=10, pady=10)

        # 创建性能指标标签
        tk.Label(metrics_frame, text="性能指标", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=5, pady=5)
        tk.Label(metrics_frame, text="验证集", font=("Arial", 12, "bold")).grid(row=0, column=1, padx=5, pady=5)
        tk.Label(metrics_frame, text="测试集", font=("Arial", 12, "bold")).grid(row=0, column=2, padx=5, pady=5)

        # Accuracy
        tk.Label(metrics_frame, text="Accuracy", font=("Arial", 10)).grid(row=1, column=0, padx=5, pady=5)
        tk.Label(metrics_frame, text="{:.2f}".format(val_accuracy), font=("Arial", 10)).grid(row=1, column=1, padx=5,
                                                                                             pady=5)
        tk.Label(metrics_frame, text="{:.2f}".format(test_accuracy), font=("Arial", 10)).grid(row=1, column=2, padx=5,
                                                                                              pady=5)

        # Precision
        tk.Label(metrics_frame, text="Precision", font=("Arial", 10)).grid(row=2, column=0, padx=5, pady=5)
        tk.Label(metrics_frame, text="{:.2f}".format(val_precision), font=("Arial", 10)).grid(row=2, column=1, padx=5,
                                                                                              pady=5)
        tk.Label(metrics_frame, text="{:.2f}".format(test_precision), font=("Arial", 10)).grid(row=2, column=2, padx=5,
                                                                                               pady=5)

        # Recall
        tk.Label(metrics_frame, text="Recall", font=("Arial", 10)).grid(row=3, column=0, padx=5, pady=5)
        tk.Label(metrics_frame, text="{:.2f}".format(val_recall), font=("Arial", 10)).grid(row=3, column=1, padx=5,
                                                                                           pady=5)
        tk.Label(metrics_frame, text="{:.2f}".format(test_recall), font=("Arial", 10)).grid(row=3, column=2, padx=5,
                                                                                            pady=5)

        # F1 Score
        tk.Label(metrics_frame, text="F1 Score", font=("Arial", 10)).grid(row=4, column=0, padx=5, pady=5)
        tk.Label(metrics_frame, text="{:.2f}".format(val_f1), font=("Arial", 10)).grid(row=4, column=1, padx=5, pady=5)
        tk.Label(metrics_frame, text="{:.2f}".format(test_f1), font=("Arial", 10)).grid(row=4, column=2, padx=5, pady=5)

        # Confusion Matrix
        # You can choose to display the confusion matrix if needed
        # 在窗口中显示混淆矩阵
        conf_matrix_frame = ttk.Frame(root)
        conf_matrix_frame.pack(padx=10, pady=10)

        conf_matrix_label = tk.Label(conf_matrix_frame, text="混淆矩阵", font=("Arial", 12, "bold"))
        conf_matrix_label.pack(pady=5)

        # 创建Canvas用于绘制混淆矩阵
        conf_matrix_canvas = tk.Canvas(conf_matrix_frame, width=300, height=300, bg="white")
        conf_matrix_canvas.pack()

        # 绘制混淆矩阵
        matrix_size = len(val_conf_matrix)
        cell_width = 300 / matrix_size
        cell_height = 300 / matrix_size

        for i in range(matrix_size):
            for j in range(matrix_size):
                x0 = i * cell_width
                y0 = j * cell_height
                x1 = x0 + cell_width
                y1 = y0 + cell_height
                conf_matrix_canvas.create_rectangle(x0, y0, x1, y1, outline="black", fill="white")
                conf_matrix_canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(val_conf_matrix[i][j]))

        root.mainloop()

    def evaluate_svm(self):
        import tkinter as tk
        from tkinter import ttk
        import pandas as pd
        import joblib
        from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
        from sklearn.model_selection import train_test_split

        # Create a Tkinter window
        root = tk.Tk()

        title_label = tk.Label(root, text="SVM1算法模型性能评估如下：", font=("Arial", 14, "bold"))
        root.title("SVM算法模型性能评估如下")

        # 加载保存的模型
        loaded_model = joblib.load('svm_model.pkl')

        # 加载测试数据
        test_data = pd.read_csv('test_dataclass.csv')

        # 提取特征和标签
        X_test = test_data.drop('Class', axis=1)
        y_test = test_data['Class']

        # 使用加载的模型进行预测
        predictions = loaded_model.predict(X_test)

        # 计算测试集上的准确率、召回率、F1值和混淆矩阵
        accuracy = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        conf_matrix = confusion_matrix(y_test, predictions)

        # Display results in Tkinter window
        results_label = tk.Label(root, text="Test Set Evaluation Results", font=("Helvetica", 16))
        results_label.pack()

        accuracy_label = tk.Label(root, text=f'Accuracy: {accuracy:.4f}')
        accuracy_label.pack()

        recall_label = tk.Label(root, text=f'Recall: {recall:.4f}')
        recall_label.pack()

        f1_label = tk.Label(root, text=f'F1 Score: {f1:.4f}')
        f1_label.pack()

        conf_matrix_label = tk.Label(root, text='Confusion Matrix:')
        conf_matrix_label.pack()

        conf_matrix_text = tk.Text(root)
        conf_matrix_text.insert(tk.END, str(conf_matrix))
        conf_matrix_text.pack()

        # 加载训练数据
        train_data = pd.read_csv('train_dataclass.csv')

        # 提取特征和标签
        X_train = train_data.drop('Class', axis=1)
        y_train = train_data['Class']

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # 使用加载的模型进行验证集上的预测
        val_predictions = loaded_model.predict(X_val)

        # 计算验证集上的准确率、召回率、F1 值和混淆矩阵
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_recall = recall_score(y_val, val_predictions, average='weighted')
        val_f1 = f1_score(y_val, val_predictions, average='weighted')
        val_conf_matrix = confusion_matrix(y_val, val_predictions)

        # Display validation set results in Tkinter window
        val_results_label = tk.Label(root, text="Validation Set Evaluation Results", font=("Helvetica", 16))
        val_results_label.pack()

        val_accuracy_label = tk.Label(root, text=f'Accuracy: {val_accuracy:.4f}')
        val_accuracy_label.pack()

        val_recall_label = tk.Label(root, text=f'Recall: {val_recall:.4f}')
        val_recall_label.pack()

        val_f1_label = tk.Label(root, text=f'F1 Score: {val_f1:.4f}')
        val_f1_label.pack()

        val_conf_matrix_label = tk.Label(root, text='Confusion Matrix:')
        val_conf_matrix_label.pack()

        val_conf_matrix_text = tk.Text(root)
        val_conf_matrix_text.insert(tk.END, str(val_conf_matrix))
        val_conf_matrix_text.pack()

        root.mainloop()

    def evaluate_logreg(self):
        import tkinter as tk
        from tkinter import ttk
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        def display_metrics(metrics_text, conf_matrix):
            root = tk.Tk()
            root.title("Logistic Regression Performance Metrics")

            # 标题
            title_label = tk.Label(root, text="逻辑回归算法模型性能评估如下：", font=("Arial", 14, "bold"))
            title_label.pack(padx=10, pady=10)

            # 性能指标表格
            metrics_frame = ttk.Frame(root)
            metrics_frame.pack(padx=10, pady=5)

            metrics_tree = ttk.Treeview(metrics_frame)
            metrics_tree["columns"] = ("Metric", "Validation Set", "Test Set")
            metrics_tree.heading("#0", text="", anchor="center")
            metrics_tree.heading("Metric", text="Metric", anchor="center")
            metrics_tree.heading("Validation Set", text="Validation Set", anchor="center")
            metrics_tree.heading("Test Set", text="Test Set", anchor="center")

            for metric, val, test in metrics_text:
                metrics_tree.insert("", "end", text="", values=(metric, val, test))

            metrics_tree.pack(fill="both", expand=True)

            # 混淆矩阵
            conf_matrix_frame = ttk.Frame(root)
            conf_matrix_frame.pack(padx=10, pady=5)

            conf_matrix_label = tk.Label(conf_matrix_frame, text="Confusion Matrix", font=("Arial", 14, "bold"))
            conf_matrix_label.pack(pady=5)

            # Create a canvas to draw the confusion matrix
            canvas = tk.Canvas(conf_matrix_frame, width=300, height=300)
            canvas.pack()

            # Calculate cell width and height
            cell_width = 300 / len(conf_matrix)
            cell_height = 300 / len(conf_matrix)

            # Draw the confusion matrix
            for i in range(len(conf_matrix)):
                for j in range(len(conf_matrix[i])):
                    x0 = j * cell_width
                    y0 = i * cell_height
                    x1 = x0 + cell_width
                    y1 = y0 + cell_height
                    canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black")
                    canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(conf_matrix[i][j]))

            root.mainloop()

        # 读取训练集和测试集数据
        train_data = pd.read_csv('train_dataclass.csv')
        test_data = pd.read_csv('test_dataclass.csv')

        # 提取特征和标签
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # 数据归一化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

        # 定义逻辑回归模型
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

        # 构建管道
        pipeline = make_pipeline(StandardScaler(), model)

        # 训练模型
        pipeline.fit(X_train, y_train)

        # 在验证集上进行预测
        y_val_pred = pipeline.predict(X_val)

        # 在测试集上进行预测
        y_test_pred = pipeline.predict(X_test_scaled)

        # 计算在验证集上的性能指标
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted')
        val_recall = recall_score(y_val, y_val_pred, average='weighted')
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        val_conf_matrix = confusion_matrix(y_val, y_val_pred)

        # 计算在测试集上的性能指标
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_conf_matrix = confusion_matrix(y_test, y_test_pred)

        # 构建性能指标文本
        metrics_text = [
            ("Accuracy", val_accuracy, test_accuracy),
            ("Precision", val_precision, test_precision),
            ("Recall", val_recall, test_recall),
            ("F1 Score", val_f1, test_f1)
        ]

        # 显示性能指标和混淆矩阵
        display_metrics(metrics_text, test_conf_matrix)


def train_models():
    your_class_instance = YourClass()
    your_class_instance.window.mainloop()

# Call the train_models function to start the GUI
train_models()
