#import pandas as pd
#file_path=r"D:\python集合\ROSE .csv"
#df=pd.read_csv(file_path)
#print(df)，显示数据表的代码

#python支持向量机，注意sklearn是机器学习常见库
#去特征名里面要去空格的话，strip每次只能处理一个
#【x.strip()for x in col],strip去除前后空格
#df.duplicated()，检查并发现重复值，分为从前向后和从后向前，返回布尔值
#df[df.duplicated()]查看重复的记录
#df.duplicated().sum()
#drop.duplicated()删除重复值,生成一个新的数据集
#df.drop_duplicateds(replace=true)
#需要连接数据库嘛？
#异常值的处理，三倍标准差，减去均值比上标准差，如果大于3.超过三倍就是异常值，建模来说就是删除

#数据清洗df.isnull查看缺失值
#df.notnull()查看不是缺失值的数据df.fillnull填补缺失值
#df.isnull().sum()
#提取文字
#df.文字[:10]含有酒店文字的前十条信息
#df['酒店等级’]df.酒店.str.extract('\d\.\d)分/5分‘，expand=false)true 返回dataframe
#提取酒店等级
#df.酒店.str.extract(' (.+) ',expand=false),想要找前面是空格后面是空格