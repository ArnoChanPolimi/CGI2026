import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import sys
import os

# 导入逻辑版 Dataset
from dataset_logic_v1 import (
    PlantDatasetLogic as PlantDataset,
    get_logic_transforms as get_transforms,
)
from model import get_plant_model

# 处理路径和日志工具
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))
from notebook.tools.logger_utils import get_logger

# ================= 🚀 i5-13500H + 32G 内存 极限压榨配置 =================
CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

logger = get_logger(log_dir=str(SAVE_DIR), log_filename="logic_v1_training.log")

# --- 性能参数调优 ---
BATCH_SIZE = 128  # 内存 32G 绰绰有余，从 32 提升到 128，大幅加速
EPOCHS = 9
START_EPOCH = 0
INITIAL_LR = (
    0.001  # Batch Size 增大后，初始学习率可以适当调高（从 0.0005 提升到 0.001）
)
# --------------------

DEVICE = torch.device("cpu")
# =================================================================


def train():
    logger.info(f"--- 🚀 开启性能压榨模式 (Batch={BATCH_SIZE}, Workers=12) ---")

    # 1. 计算权重
    df = pd.read_csv(CSV_PATH)
    train_df = df[df["split"] == "train"]
    counts = train_df["label"].value_counts().sort_index().values
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    weights = weights / weights.sum() * 2.0
    class_weights = weights.to(DEVICE)

    # 2. 准备数据加载 (极致优化版)
    train_tf, _ = get_transforms()
    train_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="train", transform=train_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,  # 针对你的 16 线程，使用 12 个作为搬砖工
        pin_memory=True,  # 内存加速
        prefetch_factor=2,  # 提前准备 2 个 batch 的数据
        persistent_workers=True,  # 减少轮次间切换的初始化时间
    )

    # 3. 初始化模型
    model = get_plant_model(num_classes=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. 核心训练循环
    for epoch in range(START_EPOCH, EPOCHS):

        # 每 3 轮学习率减半逻辑
        current_lr = INITIAL_LR * (0.5 ** (epoch // 3))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        logger.info(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr}")

        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 打印频率：由于 Batch 变大了，每 20 步打印一次即可
            if (i + 1) % 20 == 0:
                msg = f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                print(msg)
                if (i + 1) % 100 == 0:
                    logger.info(msg)

        avg_loss = running_loss / len(train_loader)
        logger.info(f"===> Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")

        # 保存权重
        save_path = SAVE_DIR / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)

    logger.info("--- 🚀 性能压榨版训练圆满完成！ ---")


if __name__ == "__main__":
    train()
