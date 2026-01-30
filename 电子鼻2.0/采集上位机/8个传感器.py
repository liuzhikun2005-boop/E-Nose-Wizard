import sys
import os
import time
import serial
import serial.tools.list_ports
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                             QComboBox, QFileDialog, QMessageBox)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
import pyqtgraph as pg
import numpy as np

# --- 串口工作线程 ---
class SerialWorker(QThread):
    data_received = pyqtSignal(list)    # 发送解析后的数据给UI
    group_finished = pyqtSignal()      # 标志一组数据结束
    error_occurred = pyqtSignal(str)    # 错误信号
    
    # 更改：传感器数量由 10 变为 8
    EXPECTED_DATA_LENGTH = 8 + 2  # Count(1) + S1-S8(8) + Flag(1) = 10

    def __init__(self):
        super().__init__()
        self.port = None
        self.baudrate = 115200
        self.is_running = False
        self.save_path = ""
        self.current_file = None
        self.last_flag = -1    # 用于检测 3 -> 0 的跳变
        self.group_counter = 0  # 文件组计数器

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            
            # 清空输入缓冲区，确保只读取连接后的最新数据
            self.ser.reset_input_buffer() 
            self.is_running = True
        except Exception as e:
            self.error_occurred.emit(str(e))
            return

        # 初始创建第一个文件
        self.create_new_file()

        while self.is_running:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue

                    line_to_write = line.replace('，', ',')
                    parts = line_to_write.split(',')

                    # 更改：检查数据长度是否符合 8 个传感器 + Count + Flag
                    if len(parts) >= self.EXPECTED_DATA_LENGTH:
                        try:
                            values = [float(x) for x in parts]
                            flag = int(values[-1])
                            
                            # === 1. 检测标志位跳变 (3 -> 0) ===
                            if self.last_flag == 3 and flag == 0:
                                # A. 关闭旧文件并打开新文件
                                self.create_new_file()    
                                # B. 通知UI清空图像
                                self.group_finished.emit() 

                            # === 2. 写入数据到文件 ===
                            if self.current_file:
                                self.current_file.write(line_to_write + '\n')
                                self.current_file.flush() # 确保实时写入

                            # === 3. 发送数据给UI绘图 ===
                            self.data_received.emit(values)

                            self.last_flag = flag

                        except ValueError:
                            pass # 数据转换失败，忽略坏点
            except Exception as e:
                self.error_occurred.emit(f"读取流错误: {e}")
                break

        # 停止后的清理
        if self.ser and self.ser.is_open:
            self.ser.close()
        if self.current_file:
            self.current_file.close()

    def create_new_file(self):
        """创建新的TXT存储文件，并更新文件名格式和组计数"""
        if not self.save_path:
            return
            
        if self.current_file:
            self.current_file.close() # 关闭旧文件

        # 1. 增加组计数
        self.group_counter += 1
        
        # 2. 生成文件名：YYYYMMDD_HHMM_GroupX.txt
        now = datetime.now()
        timestamp_prefix = now.strftime("%Y%m%d_%H%M") 
        
        filename = os.path.join(
            self.save_path, 
            f"{timestamp_prefix}_Group{self.group_counter}.txt"
        )
        
        try:
            # 打开新文件
            self.current_file = open(filename, 'w', encoding='utf-8')
            # 更改：更新表头以匹配 8 个传感器
            header = "Count,S1,S2,S3,S4,S5,S6,S7,S8,Flag\n"
            self.current_file.write(header)
        except Exception as e:
            self.error_occurred.emit(f"无法创建文件: {e}")

    def stop(self):
        """安全停止线程"""
        self.is_running = False
        self.wait()

# --- 主界面 ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多通道传感器实时采集上位机 (8通道)")
        self.resize(1200, 800)

        # 更改：核心变量，传感器数量变为 8
        self.sensor_count = 8
        self.plot_curves = [] # 存储8条曲线对象
        self.x_data = [[] for _ in range(self.sensor_count)] # 存储X轴数据 (cnt)
        self.y_data = [[] for _ in range(self.sensor_count)] # 存储Y轴数据 (sensor)
        
        self.save_directory = os.getcwd() # 默认保存路径为当前目录

        self.init_ui()
        
        self.worker = None

    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 1. 顶部控制栏
        control_layout = QHBoxLayout()
        
        # 串口选择
        self.combo_port = QComboBox()
        self.refresh_ports()
        control_layout.addWidget(QLabel("端口:"))
        control_layout.addWidget(self.combo_port)
        
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.refresh_ports)
        control_layout.addWidget(btn_refresh)

        # 波特率
        self.combo_baud = QComboBox()
        self.combo_baud.addItems(["9600", "115200", "256000", "921600"])
        self.combo_baud.setCurrentText("115200")
        control_layout.addWidget(QLabel("波特率:"))
        control_layout.addWidget(self.combo_baud)

        # 路径选择
        self.lbl_path = QLabel(f"保存路径: {self.save_directory}")
        btn_path = QPushButton("选择文件夹")
        btn_path.clicked.connect(self.select_folder)
        control_layout.addWidget(QLabel("文件保存位置:"))
        control_layout.addWidget(btn_path)
        control_layout.addWidget(self.lbl_path)

        # 开始/停止按钮
        self.btn_start = QPushButton("开始采集")
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(self.toggle_capture)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        control_layout.addWidget(self.btn_start)

        layout.addLayout(control_layout)

        # 2. 图表区域 (网格布局 4行2列)
        grid_layout = QGridLayout()
        # 配置 pyqtgraph
        pg.setConfigOption('background', 'k') # 黑色背景
        pg.setConfigOption('foreground', 'd') # 灰色前景

        for i in range(self.sensor_count):
            # 创建 PlotWidget
            plot_widget = pg.PlotWidget(title=f"传感器 {i+1}")
            plot_widget.showGrid(x=True, y=True, alpha=0.5)
            plot_widget.setLabel('left', 'Value')
            plot_widget.setLabel('bottom', 'Count')
            
            # 启用鼠标交互 (左键拖拽=平移，右键拖拽=缩放/滚轮=缩放)
            plot_widget.setMouseEnabled(x=True, y=True)
            
            # 创建曲线
            curve = plot_widget.plot(pen=pg.mkPen(color=pg.intColor(i, self.sensor_count), width=1.5)) # 不同颜色曲线
            self.plot_curves.append(curve)
            
            # 更改：添加到网格 (8个图表 -> 4行2列)
            row = i // 2
            col = i % 2
            grid_layout.addWidget(plot_widget, row, col)

        layout.addLayout(grid_layout)

    def refresh_ports(self):
        """刷新可用串口列表"""
        self.combo_port.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.combo_port.addItem(p.device)

    def select_folder(self):
        """选择文件保存路径"""
        folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if folder:
            self.save_directory = folder
            self.lbl_path.setText(folder)

    def toggle_capture(self):
        """开始/停止采集"""
        if self.btn_start.isChecked():
            # 开始
            if self.combo_port.currentText() == "":
                QMessageBox.warning(self, "错误", "未选择串口！")
                self.btn_start.setChecked(False)
                return
            if not os.path.isdir(self.save_directory):
                 QMessageBox.warning(self, "错误", "保存路径无效或不存在！")
                 self.btn_start.setChecked(False)
                 return

            self.btn_start.setText("停止采集")
            self.btn_start.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
            
            # 启动线程
            self.worker = SerialWorker()
            self.worker.port = self.combo_port.currentText()
            self.worker.baudrate = int(self.combo_baud.currentText())
            self.worker.save_path = self.save_directory
            
            self.worker.data_received.connect(self.update_graphs)
            self.worker.group_finished.connect(self.reset_graphs) # 3->0 时触发
            self.worker.error_occurred.connect(self.handle_error)
            
            self.worker.start()
            
        else:
            # 停止
            self.btn_start.setText("开始采集")
            self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            if self.worker:
                self.worker.stop()
                self.worker = None

    def update_graphs(self, values):
        """接收串口数据并更新绘图缓存"""
        cnt = values[0]
        # 更改：只取 S1 到 S8 的数据 (values[1] 到 values[8])
        sensors = values[1:self.sensor_count + 1]

        # 更新每个传感器的数据列表
        for i in range(self.sensor_count):
            self.x_data[i].append(cnt)
            self.y_data[i].append(sensors[i])
            
            # 实时更新曲线
            self.plot_curves[i].setData(self.x_data[i], self.y_data[i])

    def reset_graphs(self):
        """一组数据结束，清空图像"""
        for i in range(self.sensor_count):
            self.x_data[i] = []
            self.y_data[i] = []
            self.plot_curves[i].setData([], [])

    def handle_error(self, msg):
        """处理线程错误"""
        QMessageBox.critical(self, "错误", msg)
        self.btn_start.setChecked(False)
        self.toggle_capture() # 尝试安全停止

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())