import json
import os
import redis
import sys


absfile=os.path.abspath(__file__)
basedir=os.path.dirname(absfile)
basedir=os.path.dirname(basedir)
rootdir=os.path.dirname(basedir)
sys.path.insert(0,rootdir)
from base.config import Config
from base.log import get_logger

class redis_client:
    def __init__(self):
        try:
            self.config=Config()
            self.logger=get_logger()
            self.client = redis.StrictRedis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT, # 从 config.ini 读取（固定配置）
                #password=self.config.REDIS_PASSWORD,  # 从 .env 读取（敏感配置）
                db=self.config.REDIS_DB,  # 从 config.ini 读取（固定配置）
                decode_responses=True  # 自动将 bytes 解码为 str（类似 Java 的 StringRedisSerializer）
            )
            self.logger.info("redis连接成功")
        except Exception as e:
            self.logger.error(f"redis连接失败: {e}")
            raise

    def set_data(self, key, value):
        """
        存储数据到 Redis
        """
        try:
            # 存储 JSON 数据
            self.client.set(key, json.dumps(value, ensure_ascii=False))
            self.logger.info(f"存储数据到 KEY: {key}")
        except redis.RedisError as e:
            self.logger.error(f"Redis 存储失败: {e}")
            raise

    def get_data(self, key):
        """
        从 Redis 获取数据
        """
        try:
            value = self.client.get(key)
            if value is not None:
                return json.loads(value)  # 将 JSON 字符串转换回 Python 对象
            else:
                self.logger.warning(f"KEY: {key} 不存在")
                return None
        except redis.RedisError as e:
            self.logger.error(f"Redis 获取失败: {e}")
            raise


if __name__ == '__main__':
    redis_client=redis_client()
    data={"name":"张三","age":30}
    redis_client.set_data("user:1",data)

