import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import sys
import os
import re
from tqdm import tqdm

# 确保引用最新的 Letterbox 脚本和模型
from dataset_letterbox import (
    PlantDatasetLetterbox as PlantDataset,
    get_letterbox_transforms as get_transforms,
)
from model import get_plant_model

current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))
from notebook.tools.logger_utils import get_logger

# ================= 🚀 路径对齐：使用最新的 _order.csv =================
CSV_PATH = (
    ROOT_DIR
    / "output"
    / "dataset_index"
    / "dataset_index_letterbox_NoHealthy_v1_NoAloeVera_order.csv"
)

# 权重保存与日志路径
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1" / "pth"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = ROOT_DIR / "log" / "train" / "All_crop_train_logic_v1"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = get_logger(log_dir=str(LOG_DIR), log_filename="training_order_fast.log")
# =====================================================================

# --- 🚀 性能压榨调优 ---
BATCH_SIZE = 128  # 32G内存可以尝试 128，如果报错再降回 64
EPOCHS = 20  # 增加轮数，因为顺序划分后收敛可能变慢
INITIAL_LR = 0.001
DEVICE = torch.device("cpu")  # 确保 CPU 训练


def train():
    logger.info(f"--- 🚀 开启【极限加速】模式 ---")
    logger.info(f"Batch Size: {BATCH_SIZE} | CSV: {CSV_PATH.name}")

    # 1. 自动接力 (对齐新的文件名格式)
    all_pths = list(SAVE_DIR.glob("checkpoint_epoch_order_*.pth"))
    latest_epoch = 0
    latest_pth = None

    if all_pths:
        try:
            epochs = [
                int(re.findall(r"checkpoint_epoch_order_(\d+).pth", p.name)[0])
                for p in all_pths
            ]
            latest_epoch = max(epochs)
            latest_pth = SAVE_DIR / f"checkpoint_epoch_order_{latest_epoch}.pth"
        except:
            pass

    # 2. 类别权重计算 (应对样本不平衡)
    df = pd.read_csv(CSV_PATH)
    train_df = df[df["split"] == "train"]
    counts = train_df["label"].value_counts().sort_index().values
    class_weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    class_weights = (class_weights / class_weights.sum() * 2.0).to(DEVICE)

    logger.info(f"⚖️ 类别权重: {class_weights.tolist()}")

    # 3. 数据加载提速核心 (压榨 12 核 CPU)
    train_tf, _ = get_transforms()
    train_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="train", transform=train_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 顺序划分后，训练时必须 shuffle！
        num_workers=8,  # i5-13500H 建议设为 8-10，不要占满全部 12 核以防系统卡死
        pin_memory=False,  # CPU 训练设为 False
        prefetch_factor=3,  # 增加预取倍数
        persistent_workers=True,  # 保持 worker 进程，避免每轮重新创建
    )

    model = get_plant_model(num_classes=2).to(DEVICE)
    if latest_pth and latest_pth.exists():
        logger.info(f"🔄 加载历史权重: {latest_pth.name}")
        model.load_state_dict(torch.load(latest_pth, map_location=DEVICE))
        start_from = latest_epoch
    else:
        start_from = 0

    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. 训练循环
    for epoch in range(start_from, EPOCHS):
        # 每 5 轮衰减一次学习率
        current_lr = INITIAL_LR * (0.5 ** (epoch // 5))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        model.train()
        running_loss = 0.0
        pbar = tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"
        )

        for i, (images, labels) in pbar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 20 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # 保存权重
        save_path = SAVE_DIR / f"checkpoint_epoch_order_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)

    logger.info("--- 🚀 训练圆满完成！ ---")


if __name__ == "__main__":
    train()
