import logging
import os

absfile=os.path.abspath(__file__)
basedir=os.path.dirname(absfile)
rootdir=os.path.dirname(basedir)

log_path=os.path.join(rootdir,'app/app.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)
#创建logger对象
def get_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    #创建handler对象
    sh=logging.StreamHandler()
    sh.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding='utf-8', mode='a')
    fh.setLevel(logging.WARN)

    #创建formatter对象
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    #将formatter添加到handler中
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    #将handler添加到logger中
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
if __name__ == '__main__':
    logger = get_logger()
    logger.info("这是一个日志测试")
    logger.warning("这是一个警告测试")
    logger.error("这是一个错误测试")