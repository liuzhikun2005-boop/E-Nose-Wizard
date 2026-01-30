import os
import shutil

input_folder = r"E:\电子鼻\数据采集\醋"   # TXT 文件夹
output_folder = r"E:\电子鼻\数据采集\醋"  # 输出 CSV 文件夹

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        txt_path = os.path.join(input_folder, filename)

        # 更改扩展名 .txt → .csv
        csv_filename = filename.replace(".txt", ".csv")
        csv_path = os.path.join(output_folder, csv_filename)

        print(f"正在转换：{filename} → {csv_filename}")

        # 读取 TXT 文件内容
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 替换首行 Count → timestamp
        if lines and lines[0].startswith("Count"):
            lines[0] = lines[0].replace("Count", "timestamp", 1)

        # 写入 CSV 文件
        with open(csv_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

print("\n全部 TXT 已成功转换为 CSV 并修改首行！")
