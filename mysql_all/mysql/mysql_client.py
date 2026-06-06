import pandas#处理csv文件的库
import pymysql#连接mysql数据库的库
import redis#连接redis数据库的库
from base.config import Config
from base.log import get_logger
class mysql_client:
    # ① 定义 MySQL 客户端类，负责连接和操作 MySQL 数据库
    def __init__(self):
        self.config=Config()
        self.logger= get_logger()
        self.mysql_host=self.config.MYSQL_HOST
        self.mysql_port=self.config.MYSQL_PORT
        self.mysql_user=self.config.MYSQL_USER
        self.mysql_password=self.config.MYSQL_PASSWORD
        self.mysql_database=self.config.MYSQL_DATABASE
        try:
            # ② 手动创建 MySQL 连接
            # pymysql.connect() 相当于 Java 中的 DriverManager.getConnection()
            # 【关键区别】：Java 通过 application.yml 自动配置，Python 必须通过 Config 类手动读取参数
            self.connection = pymysql.connect(
                host=self.mysql_host,
                user=self.mysql_user,
                password=self.mysql_password,  # 从 .env 读取（敏感配置）
                database=self.mysql_database # 从 config.ini 读取（固定配置）
            )
            # ③ 创建游标对象（Cursor）用于执行 SQL 语句
            self.cursor = self.connection.cursor()
            self.logger.info("MySQL 连接成功")
        except Exception as e:
            self.logger.error(f"MySQL 连接失败: {e}")
            raise   # 连接失败时抛出异常，阻止继续执行

    # 定义创建表的方法，执行标准的 MySQL 建表语句
    def create_table(self):
        # ======== 第一步：编写 SQL 语句 ========
        # 用 Python 三引号 ''' 定义多行字符串，里面写标准的 MySQL 建表语句
        # 这个 SQL 语句和你在 MySQL 命令行 / Navicat 中写的完全一样
        create_table_query = """ CREATE TABLE IF NOT EXISTS jpkb (
            id INT AUTO_INCREMENT PRIMARY KEY,
            subject_name VARCHAR(20),
            question VARCHAR(1000),
            answer VARCHAR(1000))
        """
        try:
            # ======== 第二步：通过游标执行 SQL ========
            # cursor.execute() 是所有 SQL 操作的统一入口
            # 无论是 CREATE、INSERT、SELECT、UPDATE、DELETE，都通过这个方法执行
            self.cursor.execute(create_table_query)

            # ======== 第三步：提交事务 ========
            self.connection.commit()
            self.logger.info("表创建成功")

        except pymysql.MySQLError as e:
            self.logger.error(f"表创建失败: {e}")
            raise  # 建表失败是严重错误，必须抛出让调用方知道

    #定义插入数据的方法，读取 CSV 文件并执行 INSERT 语句
    def insert_data(self,csv_file):
        try:
            data = pandas.read_csv(csv_file)
            # 通过 iterrows() 方法遍历 DataFrame 的(索引，每一行）
            for index, row in data.iterrows():
                subject_name = row['学科名称']
                question = row['问题']
                answer = row['答案']
                insert_query = "INSERT INTO jpkb (subject_name, question, answer) VALUES (%s, %s, %s)"
                #使用游标进行操作，执行 SQL 语句并传入参数
                self.cursor.execute(insert_query, (subject_name, question, answer))
                self.connection.commit()
        except Exception as e:
            self.logger.error(f"数据插入失败: {e}")
            #获得connection对象，用来获取游标和提交事务
            self.connection.rollback()  # 插入失败时回滚事务，保持数据一致性
            raise

    #查询所有数据的方法，执行 SELECT 语句并返回结果
    def query_data(self):
        try:
            query = "SELECT subject_name,question, answer FROM jpkb"
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            self.logger.info(f"查询数据成功")
            return results
        except Exception as e:
            self.logger.error(f"数据查询失败: {e}")
            raise

    #根据问题查询数据的方法，执行带 WHERE 条件的 SELECT 语句并返回结果
    def query_qustion(self,question):
        try:
            query = "SELECT subject_name,question, answer FROM jpkb WHERE question=%(question)s"
            self.cursor.execute(query,{"question":question})
            results = self.cursor.fetchall()
            self.logger.info(f"查询数据成功")
            return results
        except Exception as e:
            self.logger.error(f"数据查询失败: {e}")
            raise

    # 定义关闭连接的方法，释放资源
    def close(self):
        try:
            self.connection.close()
            self.logger.info("MySQL 连接已关闭")
        except Exception as e:
            self.logger.error(f"关闭 MySQL 连接失败: {e}")
            raise

if __name__ == '__main__':
    mysql_client=mysql_client()
    mysql_client.create_table()
    mysql_client.insert_data(r"E:\PythonProject2\bs25_mysql_redis\data\JP学科知识问答.csv")
    mysql_client.logger.info("数据插入成功")
    results=mysql_client.query_data()
    for result in results:
        print(result)

    mysql_client.close()




