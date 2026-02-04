import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import sys
import os
from tqdm import tqdm  # <--- 必须导入这个

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
BATCH_SIZE = 128
EPOCHS = 9
START_EPOCH = 0
INITIAL_LR = 0.001
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

    # 2. 准备数据加载
    train_tf, _ = get_transforms()
    train_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="train", transform=train_tf)

    # 【压榨建议】如果你发现运行报错 "BrokenPipeError"，请将 num_workers 降为 8 或 4
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # 3. 初始化模型
    model = get_plant_model(num_classes=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. 核心训练循环
    for epoch in range(START_EPOCH, EPOCHS):
        current_lr = INITIAL_LR * (0.5 ** (epoch // 3))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        logger.info(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr}")
        model.train()
        running_loss = 0.0

        # --- 🚀 注入 tqdm 进度条 ---
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}",
            unit="batch",
        )

        for i, (images, labels) in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # --- 进度条实时交互 ---
            if (i + 1) % 10 == 0:
                # 在进度条右侧实时显示 Loss
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"}
                )

            # 定时向日志文件写入，不影响终端显示
            if (i + 1) % 100 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        logger.info(f"===> Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")

        # 保存权重
        save_path = SAVE_DIR / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"权重已保存: {save_path.name}")

    logger.info("--- 🚀 性能压榨版训练圆满完成！ ---")


if __name__ == "__main__":
    # Windows 环境下建议增加此保护
    train()
