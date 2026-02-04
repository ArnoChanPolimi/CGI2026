import logging
import os
from datetime import datetime


def get_logger(name="train_log"):
    # 创建 log 文件夹
    if not os.path.exists("log"):
        os.makedirs("log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler
    if not logger.handlers:
        # 命名规则：log/20260120_1530.log
        log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
        log_path = os.path.join("log", log_filename)

        # 文件输出
        fh = logging.FileHandler(log_path, encoding="utf-8")
        # 屏幕输出
        ch = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
