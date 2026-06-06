import sys
import os
import time

absfile=os.path.abspath(__file__)
dirfile=os.path.dirname(absfile)
sys.path.insert(0,dirfile)
from cache.redis_client import redis_client
from mysql.mysql_client import mysql_client
from retrieval.bm25_search import BM25Search
from base.log import get_logger
class Main:
    # 初始化方法，创建mysql和redis的连接对象，以及BM25搜索对象
    def __init__(self):
        self.logger = get_logger()
        self.mysql_client = mysql_client()
        self.redis_client = redis_client()
        self.bm25_search = BM25Search(self.mysql_client, self.redis_client)
    # 主方法，执行搜索流程
    def query(self, query):
        self.logger.info(f"用户查询: {query}")
        start=time.time()
        answer,_=self.bm25_search.search(query,0.7)
        if answer is not None:
            self.logger.info(f"查询结果: {answer}")
        else:
            self.logger.info("未找到相关答案")
        end=time.time()
        self.logger.info(f"查询耗时: {end - start:.4f} 秒")
    def main(self):
        while True:
            query = input("请输入查询内容（输入 'exit' 退出）：")
            if query.lower() == 'exit':
                self.logger.info("用户退出程序")
                break
            self.query(query)

if __name__ == '__main__':
    main = Main()
    main.main()



