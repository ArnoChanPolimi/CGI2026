# notebook\train\train_main_logic_v1.py
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

    # 1. 自动扫描最新的 pth 文件 (核心增强)
    all_pths = list(SAVE_DIR.glob("checkpoint_epoch_*.pth"))
    latest_epoch = 0
    latest_pth = None

    if all_pths:
        # 提取文件名里的数字并找出最大的
        epochs = [
            int(re.findall(r"checkpoint_epoch_(\[0-9\]+).pth", p.name)[0])
            for p in all_pths
        ]
        latest_epoch = max(epochs)
        latest_pth = SAVE_DIR / f"checkpoint_epoch_{latest_epoch}.pth"

    # 2. 初始化模型
    model = get_plant_model(num_classes=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)

    # 3. 加载逻辑
    if latest_pth and latest_pth.exists():
        logger.info(f"🔄 发现最新大脑：{latest_pth.name}，正在接力训练...")
        model.load_state_dict(torch.load(latest_pth, map_location=DEVICE))
        start_from = latest_epoch  # 从下一轮开始
        logger.info(
            f"✅ 已加载第 {latest_epoch} 轮权重，将从第 {start_from + 1} 轮继续。"
        )
    else:
        logger.info("🆕 未发现历史权重，将从官方预训练大脑开始训练。")
        start_from = 0

    model = get_plant_model(num_classes=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(start_from, EPOCHS):
        # 学习率逻辑依然保持每 3 轮减半
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
