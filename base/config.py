import configparser
import os
from dotenv import load_dotenv
import sys
absfile=os.path.abspath(__file__)
basedir=os.path.dirname(absfile)
sys.path.insert(0,basedir)
from log import get_logger

logger = get_logger()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

class Config:
    def __init__(self):
        try:
            self.config = configparser.ConfigParser()
            #对于固定配置ini，先获取对象，再获取路径，最后对象.read读取文件。对象.get获取值
            #只有在同一级目录下才可以直接使用文件名，否则需要提供完整路径
            absfile=os.path.abspath(__file__)
            basedir=os.path.dirname(absfile)
            rootdir=os.path.dirname(basedir)
            env_path=os.path.join(rootdir,'config.ini')
            self.config.read(env_path, encoding='utf-8')
            self.MYSQL_HOST=self.config.get("mysql","localhost")
            self.MYSQL_PORT=self.config.getint("mysql","port")
            self.MYSQL_USER=self.config.get("mysql","user")
            self.MYSQL_DATABASE=self.config.get("mysql","database")
            self.MYSQL_USER=self.config.get("mysql","user")
            self.REDIS_HOST=self.config.get("cache","localhost")
            self.REDIS_PORT=self.config.getint("cache","port")
            self.REDIS_DB=self.config.getint("cache","db")
            self.MODEL=self.config.get("llm","model")
            self.URL=self.config.get("llm","base_url")

            self.REDIS_PASSWORD=os.getenv("REDIS_PASSWORD","123456")
            self.MYSQL_PASSWORD=os.getenv("MYSQL_PASSWORD","123456")
            self.DEEPSEEK_APIKEY=os.getenv("DEEPSEEK_API_KEY","")

            self.PARENT_CHUNK_SIZE=self.config.getint("retrieval","parent_chunk_size")
            self.CHILD_CHUNK_SIZE=self.config.getint("retrieval","child_chunk_size")
            self.CHUNK_OVERLAP=self.config.getint("retrieval","chunk_overlap")
            self.RETRIEVAL_K=self.config.getint("retrieval","retrieval_k")
            self.CONDIDATE_M=self.config.getint("retrieval","candidate_m")



            self.COLLECTION_NAME = self.config.get("milvus","collection_name")
            self.MILVUS_HOST = self.config.get("milvus","host")
            self.MILVUS_PORT = self.config.getint("milvus","port")
            self.MILVUS_DATABASE = self.config.get("milvus","database")

            logger.info("config：配置文件读取成功")
        except Exception as e:

            logger.error(f"配置文件读取失败: {e}")
            raise


if __name__ == '__main__':
    logger.info("这是一个日志测试")
    config=Config()
    print(config.MYSQL_HOST)
    print(config.REDIS_HOST)



