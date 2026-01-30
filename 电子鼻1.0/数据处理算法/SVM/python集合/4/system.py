import tkinter as tk
from tkinter import font

class SmellClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多通道气体传感器的气味分类系统")
        self.root.geometry("700x400")
        # 加载背景图片
        self.background_image = tk.PhotoImage(file="back.png")  # 更改为你的背景图片的文件路径

        # 在整个窗口上放置背景图片
        self.background_label = tk.Label(root, image=self.background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # 定义字体样式
        self.custom_font = font.Font(family='黑体', size=14)

        # 左侧功能按钮，应用自定义字体
        self.button_data_collection = tk.Button(root, text="数据采集", command=self.collect_data, width=20, height=2, font=self.custom_font)
        self.button_data_collection.place(x=50, y=40)

        self.button_data_monitoring = tk.Button(root, text="数据监控", command=self.monitor_data, width=20, height=2, font=self.custom_font)
        self.button_data_monitoring.place(x=50, y=100)

        self.button_pca_visualization = tk.Button(root, text="PCA降维可视化", command=self.visualize_pca, width=20, height=2, font=self.custom_font)
        self.button_pca_visualization.place(x=50, y=160)

        self.button_model_training = tk.Button(root, text="机器学习训练", command=self.train_models, width=20, height=2, font=self.custom_font)
        self.button_model_training.place(x=50, y=220)

        self.button_smell_classification = tk.Button(root, text="气味分类", command=self.classify_smell, width=20, height=2, font=self.custom_font)
        self.button_smell_classification.place(x=50, y=280)
        self.label_welcome = tk.Label(root, text="欢迎来到气味分类系统！", font=("黑体", 20, "bold"))

        self.label_welcome.place(x=190, y=1)
        # 右侧背景图片
        self.canvas = tk.Canvas(root, width=390, height=235, bg='lightblue')
        self.canvas.place(x=300, y=50)
        self.background_image_right = tk.PhotoImage(file="background.png")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background_image_right)

    def collect_data(self):
        import serial
        import mysql.connector
        import tkinter as tk

        # Function to handle start button click
        def start():
            global ser, mycursor, mydb

            # Configure serial port
            ser = serial.Serial('COM5', 115200)  # Modify the port and baud rate as per your setup

            # Configure MySQL connection
            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password="dir99",
                database="nose")
            mycursor = mydb.cursor()

            # Start reading and processing serial data
            read_serial_data()

        # Function to handle stop button click
        def stop():
            global ser, mycursor, mydb

            # Close serial port and database connection
            ser.close()
            mycursor.close()
            mydb.close()

        # Function to read and process serial data
        def read_serial_data():
            try:
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode().strip()
                        print("Received:", data)

                        if data.startswith('DATA,'):
                            values = data.split(',')[1:]
                            if len(values) == 4:
                                sql = "INSERT INTO dianzibi (GM102B, GM302B, GM502B, GM702B, CLASS) VALUES (%s, %s, %s, %s, %s)"
                                class_value = 0
                                val = (values[0], values[1], values[2], values[3], class_value)
                                mycursor.execute(sql, val)
                                mydb.commit()
                                mycursor.execute("SELECT * FROM dianzibi")
                                print("Table dianzibi:")
                                for row in mycursor.fetchall():
                                    print(row)

            except KeyboardInterrupt:
                print("Exiting program...")

        # Create Tkinter window
        window = tk.Tk()
        window.title("Serial Data Logger")

        # Start button
        start_button = tk.Button(window, text="Start", command=start)
        start_button.pack()

        # Stop button
        stop_button = tk.Button(window, text="Stop", command=stop)
        stop_button.pack()

        window.mainloop()
    def monitor_data(self):
        import tkinter as tk
        from tkinter import ttk
        import serial
        import threading
        import numpy as np
        from collections import deque
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import time

        # 初始化串口
        ser = serial.Serial('COM5', 115200)  # 根据实际串口情况修改

        # 创建GUI窗口
        root = tk.Tk()
        root.title("Real-time Data Visualization")

        # 创建Frame来放置控件
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.TOP, padx=10, pady=10)

        # 创建开始和停止按钮
        start_button = ttk.Button(control_frame, text="Start", width=10)
        stop_button = ttk.Button(control_frame, text="Stop", width=10)

        # 将按钮放置到Frame中
        start_button.grid(row=0, column=0, padx=5, pady=5)
        stop_button.grid(row=0, column=1, padx=5, pady=5)

        # 创建一个画布用于显示实时数据
        figure = plt.figure(figsize=(8, 4))
        ax = figure.add_subplot(111)
        ax.set_xlabel("Time")
        ax.set_ylabel("Data")
        line1, = ax.plot([], [], color='red', label='NO2')
        line2, = ax.plot([], [], color='blue', label='C2H5CH')
        line3, = ax.plot([], [], color='yellow', label='VOC')
        line4, = ax.plot([], [], color='green', label='CO')
        ax.legend()
        canvas = FigureCanvasTkAgg(figure, master=root)
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # 用于存储数据的队列
        data_queue = deque(maxlen=1000)
        time_queue = deque(maxlen=1000)


        # 读取串口数据的函数
        def read_serial():
            while True:
                if ser.in_waiting > 0:
                    serial_data = ser.readline().decode().strip()
                    if serial_data.startswith("DATA"):
                        features = list(map(float, serial_data.split(',')[1:]))  # 从第二个逗号后开始解析数据
                        data_queue.append(features)
                        time_queue.append(time.time())

        # 更新实时图形的函数
        def update_graph():
            if data_queue:
                data = np.array(data_queue)
                time_data = np.array(time_queue)
                line1.set_data(time_data, data[:, 0])
                line2.set_data(time_data, data[:, 1])
                line3.set_data(time_data, data[:, 2])
                line4.set_data(time_data, data[:, 3])
                ax.relim()
                ax.autoscale_view(True, True, True)
                canvas.draw()
            root.after(1000, update_graph)

        # 开始按钮的点击事件处理函数
        def start_reading():
            # 禁用开始按钮
            start_button.config(state=tk.DISABLED)

            # 启动一个新线程来读取串口数据
            serial_thread = threading.Thread(target=read_serial)
            serial_thread.daemon = True  # 设置线程为守护线程，使其能够在主程序退出时自动关闭
            serial_thread.start()

            # 开始更新图形
            update_graph()

        # 绑定开始按钮的点击事件
        start_button.config(command=start_reading)

        # 运行主循环
        root.mainloop()

    def visualize_pca(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, roc_curve, auc
        from sklearn.decomposition import PCA
        import pandas as pd

        # 读取训练集和测试集数据
        train_data = pd.read_csv('train_dataclass.csv')
        test_data = pd.read_csv('test_dataclass.csv')

        # 提取特征和标签
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # 归一化处理数据
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)

        # PCA降维
        pca = PCA(n_components=3)
        X_train_pca = pca.fit_transform(X_train_normalized)
        X_test_pca = pca.transform(X_test_normalized)

        # 将标签转换为独热编码
        def one_hot_encode(labels, num_classes):
            num_labels = labels.shape[0]
            index_offset = np.arange(num_labels) * num_classes
            encoded_labels = np.zeros((num_labels, num_classes))
            encoded_labels.flat[index_offset + labels.ravel()] = 1
            return encoded_labels

        num_classes = len(np.unique(y_train))
        y_train_encoded = one_hot_encode(y_train, num_classes)

        # 定义逻辑回归模型
        class LogisticRegression:
            def __init__(self, input_dim, output_dim, learning_rate=0.01):
                self.W = np.random.randn(input_dim, output_dim)
                self.b = np.zeros((1, output_dim))
                self.learning_rate = learning_rate

            def softmax(self, z):
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                return exp_z / np.sum(exp_z, axis=1, keepdims=True)

            def cross_entropy_loss(self, y_pred, y_true):
                m = y_pred.shape[0]
                loss = -1 / m * np.sum(
                    y_true * np.log(y_pred + 1e-9))  # Adding a small epsilon to prevent division by zero
                return loss

            def train(self, X, y_true, epochs):
                losses = []
                for epoch in range(epochs):
                    # Forward pass
                    z = np.dot(X, self.W) + self.b
                    y_pred = self.softmax(z)

                    # Compute loss
                    loss = self.cross_entropy_loss(y_pred, y_true)
                    losses.append(loss)

                    # Backpropagation
                    dz = y_pred - y_true
                    dw = np.dot(X.T, dz)
                    db = np.sum(dz, axis=0, keepdims=True)

                    # Update parameters
                    self.W -= self.learning_rate * dw
                    self.b -= self.learning_rate * db

                return losses

        # 训练模型
        input_dim = X_train_pca.shape[1]
        output_dim = num_classes
        model = LogisticRegression(input_dim, output_dim)
        losses = model.train(X_train_pca, y_train_encoded, epochs=1000)

        # 预测并计算准确率
        def predict(X, model):
            z = np.dot(X, model.W) + model.b
            y_pred = model.softmax(z)
            return np.argmax(y_pred, axis=1)

        y_pred_train = predict(X_train_pca, model)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        print("Training Accuracy:", accuracy_train)

        y_pred_test = predict(X_test_pca, model)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        print("Test Accuracy:", accuracy_test)




        # 可视化降维后的数据
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Add more colors if needed
        markers = ['o', '^', 's', 'x', 'D', 'P', '*']  # Add more markers if needed
        for i in range(num_classes):
            ax.scatter(X_train_pca[y_train == i, 0], X_train_pca[y_train == i, 1], X_train_pca[y_train == i, 2],
                       c=colors[i % len(colors)], marker=markers[i % len(markers)], label=str(i))
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('PCA Visualization')
        plt.legend()
        plt.show()

    def train_models(self):
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
                X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2,
                                                                  random_state=42)

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
                tk.Label(metrics_frame, text="性能指标", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=5,
                                                                                          pady=5)
                tk.Label(metrics_frame, text="验证集", font=("Arial", 12, "bold")).grid(row=0, column=1, padx=5, pady=5)
                tk.Label(metrics_frame, text="测试集", font=("Arial", 12, "bold")).grid(row=0, column=2, padx=5, pady=5)

                # Accuracy
                tk.Label(metrics_frame, text="Accuracy", font=("Arial", 10)).grid(row=1, column=0, padx=5, pady=5)
                tk.Label(metrics_frame, text="{:.2f}".format(val_accuracy), font=("Arial", 10)).grid(row=1, column=1,
                                                                                                     padx=5,
                                                                                                     pady=5)
                tk.Label(metrics_frame, text="{:.2f}".format(test_accuracy), font=("Arial", 10)).grid(row=1, column=2,
                                                                                                      padx=5,
                                                                                                      pady=5)

                # Precision
                tk.Label(metrics_frame, text="Precision", font=("Arial", 10)).grid(row=2, column=0, padx=5, pady=5)
                tk.Label(metrics_frame, text="{:.2f}".format(val_precision), font=("Arial", 10)).grid(row=2, column=1,
                                                                                                      padx=5,
                                                                                                      pady=5)
                tk.Label(metrics_frame, text="{:.2f}".format(test_precision), font=("Arial", 10)).grid(row=2, column=2,
                                                                                                       padx=5,
                                                                                                       pady=5)

                # Recall
                tk.Label(metrics_frame, text="Recall", font=("Arial", 10)).grid(row=3, column=0, padx=5, pady=5)
                tk.Label(metrics_frame, text="{:.2f}".format(val_recall), font=("Arial", 10)).grid(row=3, column=1,
                                                                                                   padx=5,
                                                                                                   pady=5)
                tk.Label(metrics_frame, text="{:.2f}".format(test_recall), font=("Arial", 10)).grid(row=3, column=2,
                                                                                                    padx=5,
                                                                                                    pady=5)

                # F1 Score
                tk.Label(metrics_frame, text="F1 Score", font=("Arial", 10)).grid(row=4, column=0, padx=5, pady=5)
                tk.Label(metrics_frame, text="{:.2f}".format(val_f1), font=("Arial", 10)).grid(row=4, column=1, padx=5,
                                                                                               pady=5)
                tk.Label(metrics_frame, text="{:.2f}".format(test_f1), font=("Arial", 10)).grid(row=4, column=2, padx=5,
                                                                                                pady=5)

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
                X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2,
                                                                  random_state=42)

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

    def classify_smell(self):

        import tkinter as tk
        from tkinter import ttk, messagebox
        import joblib
        import numpy as np

        # 忽略特定警告
        import warnings

        warnings.filterwarnings('ignore',
                                message="X does not have valid feature names, but SVC was fitted with feature names")

        # 加载保存的模型
        best_svm_model = joblib.load('svm_model.pkl')

        def classify_input():
            # 获取用户输入
            user_input = entry.get()

            # 将输入字符串转换为numpy数组
            input_values = np.array([float(x) for x in user_input.split()]).reshape(1, -1)

            # 使用加载的模型进行预测
            prediction = best_svm_model.predict(input_values)

            # 弹出消息框显示预测的类别
            messagebox.showinfo("预测结果", f'预测的类别: {prediction[0]}')

        # 创建主窗口
        root = tk.Tk()
        root.title("气味分类器")

        # 设置窗口大小和位置
        root.geometry("400x200+200+200")

        # 创建ttk样式对象
        style = ttk.Style()

        # 设置渐变色背景
        style.configure('TFrame', background='#003366')  # 设置渐变色背景，这是蓝色的渐变

        # 设置ttk样式外观
        style.theme_use('clam')

        # 添加标题标签
        title_label = ttk.Label(root, text="欢迎使用气味分类器", font=('Arial', 18), foreground="white",
                                background='#003366')
        title_label.pack(pady=10)

        # 添加标签和输入框
        label = ttk.Label(root, text="请输入4个特征值，用空格分隔:", font=('Arial', 12), foreground="white",
                          background='#003366')
        label.pack()
        entry = ttk.Entry(root, font=('Arial', 12))
        entry.pack(pady=5)

        # 添加按钮
        button = ttk.Button(root, text="进行分类", command=classify_input)
        button.pack(pady=10)

        # 运行主事件循环
        root.mainloop()


root = tk.Tk()
app = SmellClassifierApp(root)
root.mainloop()
