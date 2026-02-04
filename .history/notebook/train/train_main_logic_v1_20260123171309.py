import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import sys

# 【核心改动：导入逻辑版 Dataset 和 Transform】
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

# ================= 🚀 逻辑版训练配置 (9 Epochs + LR Decay) =================
CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
# 更改保存目录，避免覆盖旧版本
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 日志文件名也做区分
logger = get_logger(log_dir=str(SAVE_DIR), log_filename="logic_v1_training.log")

BATCH_SIZE = 32
EPOCHS = 9  # 你要求的 9 轮
START_EPOCH = 0  # 逻辑版必须从 0 开始重新学习
INITIAL_LR = 0.0005

DEVICE = torch.device("cpu")
# =================================================================


def train():
    logger.info("--- 🚀 开始逻辑版训练任务 (等比例缩放 + 9轮 + LR自动衰减) ---")

    # 1. 计算权重 (保留原有功能)
    df = pd.read_csv(CSV_PATH)
    train_df = df[df["split"] == "train"]
    counts = train_df["label"].value_counts().sort_index().values
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    weights = weights / weights.sum() * 2.0
    class_weights = weights.to(DEVICE)

    logger.info(f"类别统计: Abiotic={counts[0]}, Biotic={counts[1]}")
    logger.info(f"损失函数补偿权重: {class_weights}")

    # 2. 准备数据加载 (使用 logic 版本)
    train_tf, _ = get_transforms()
    train_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="train", transform=train_tf)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    # 3. 初始化模型
    model = get_plant_model(num_classes=2).to(DEVICE)

    # 权重加载逻辑 (保留功能，但因为是新实验，默认不加载旧的)
    checkpoint_path = SAVE_DIR / f"checkpoint_epoch_{START_EPOCH}.pth"
    if START_EPOCH > 0 and checkpoint_path.exists():
        logger.info(f"🔍 发现断点，正在加载: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        logger.info("🆕 开始全新逻辑版训练，不使用旧版畸变权重。")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)

    # 4. 核心训练循环
    for epoch in range(START_EPOCH, EPOCHS):

        # --- 【实现每 3 轮学习率减半逻辑】 ---
        # 计算公式：每过 3 轮，LR 乘以 0.5 的 (epoch // 3) 次方
        # Epoch 0,1,2 -> INITIAL_LR
        # Epoch 3,4,5 -> INITIAL_LR * 0.5
        # Epoch 6,7,8 -> INITIAL_LR * 0.25
        current_lr = INITIAL_LR * (0.5 ** (epoch // 3))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        logger.info(f"第 {epoch+1}/{EPOCHS} 轮开始，当前学习率设定为: {current_lr}")

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

            if (i + 1) % 100 == 0:
                msg = f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                print(msg)
                if (i + 1) % 500 == 0:
                    logger.info(msg)

        # 收尾动作
        avg_loss = running_loss / len(train_loader)
        logger.info(f"===> Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")

        # 保存本轮权重
        save_path = SAVE_DIR / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"模型权重已保存: {save_path.name}")

    logger.info("--- 🚀 9 轮逻辑版训练任务全部圆满完成！ ---")


if __name__ == "__main__":
    train()
