import mysql.connector

# 配置MySQL连接
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="dir99",
    database="nose"
)

# 创建游标对象
mycursor = mydb.cursor()

# 执行查询
mycursor.execute("SELECT * FROM dainzibi")

# 检索所有行
rows = mycursor.fetchall()

# 显示表格数据
for row in rows:
    print(row)

# 关闭游标和数据库连接
mycursor.close()
mydb.close()
