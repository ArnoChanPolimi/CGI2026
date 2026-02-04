import logging
import os


def get_logger(log_dir, log_filename="main_train.log", name="train_log"):
    """
    Args:
        log_dir (str/Path): 日志存放的目录 (如 output/All_crop_train)
        log_filename (str): 日志文件的名称 (如 train_v1.log)
        name (str): Logger 对象在程序内部的唯一标识名
    """
    # 1. 确保目标文件夹存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. 构建完整的日志文件路径
    log_full_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 防止重复添加 Handler (这是 Logger 逻辑中最关键的检查)
    if not logger.handlers:
        # 3. 创建文件处理器 (FileHandler)，mode='a' 开启追加模式
        fh = logging.FileHandler(log_full_path, mode="a", encoding="utf-8")

        # 4. 创建控制台处理器 (StreamHandler)
        ch = logging.StreamHandler()

        # 5. 设置统一的格式
        # %(asctime)s: 时间, %(levelname)s: 级别(INFO/ERROR等), %(message)s: 日志内容
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 6. 将处理器添加到 logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        # 禁止日志向上级传递，避免在某些环境下（如 Notebook）出现双重打印
        logger.propagate = False

    return logger
