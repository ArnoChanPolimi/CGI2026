# notebook\train\train_main_logic_v1.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import sys
import os
import re  # <--- 修正 1: 必须导入 re
from tqdm import tqdm

# 导入逻辑版 Dataset
# from dataset_logic_v1 import (
#     PlantDatasetLogic as PlantDataset,
#     get_logic_transforms as get_transforms,
# )
# 逻辑：必须引用你刚写好的 Letterbox 脚本，才能实现“长边保真缩放”
from dataset_letterbox import (
    PlantDatasetLetterbox as PlantDataset,
    get_letterbox_transforms as get_transforms,
)
from model import get_plant_model

# 处理路径和日志工具
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))
from notebook.tools.logger_utils import get_logger

# ================= 🚀 i5-13500H + 32G 内存 极限压榨配置 =================
# CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
# CSV_PATH = (
#     ROOT_DIR / "output" / "dataset_index" / "dataset_index_letterbox_v1.csv"
# )  # <--- 对齐新 CSV 名称
# train_main_logic_v1.py 里的配置区
CSV_PATH = (
    ROOT_DIR / "output" / "dataset_index" / "dataset_index_letterbox_NoHealthy_v1.csv"
)
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

logger = get_logger(
    log_dir=str(SAVE_DIR), log_filename="logic_v1_training_letterbox.log"
)

# --- 性能参数调优 ---
BATCH_SIZE = 128
EPOCHS = 12  # 9轮可能不够，适当增加
INITIAL_LR = 0.001
DEVICE = torch.device("cpu")
# =================================================================
logger.info(f"======================================================")


def train():
    logger.info(f"--- 🚀 开启性能压榨模式 (Batch={BATCH_SIZE}, Workers=12) ---")

    # 1. 自动扫描最新的 pth 文件
    all_pths = list(SAVE_DIR.glob("checkpoint_epoch_letterbox_*.pth"))
    latest_epoch = 0
    latest_pth = None

    if all_pths:
        try:
            # 修正 2: 修正正则表达式转义
            epochs = [
                int(re.findall(r"checkpoint_epoch_letterbox_(\d+).pth", p.name)[0])
                for p in all_pths
            ]
            latest_epoch = max(epochs)
            latest_pth = SAVE_DIR / f"checkpoint_epoch_letterbox_{latest_epoch}.pth"
        except Exception as e:
            logger.warning(f"扫描权重文件名出错: {e}，将从头开始。")

    # 2. 计算权重 (类别补偿)
    df = pd.read_csv(CSV_PATH)
    train_df = df[df["split"] == "train"]
    counts = train_df["label"].value_counts().sort_index().values
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    weights = weights / weights.sum() * 2.0
    class_weights = weights.to(DEVICE)
    # 在 train() 函数里加入
    logger.info(f"📊 类别分布统计: Biotic={counts[0]}, Abiotic={counts[1]}")
    logger.info(f"⚖️ 最终计算权重: {class_weights.tolist()}")

    # 3. 准备数据加载 (修正 3: 补全之前漏掉的数据加载代码)
    train_tf, _ = get_transforms()
    train_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="train", transform=train_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # 4. 初始化模型与加载逻辑
    model = get_plant_model(num_classes=2).to(DEVICE)

    if latest_pth and latest_pth.exists():
        logger.info(f"🔄 发现最新大脑：{latest_pth.name}，正在接力训练...")
        model.load_state_dict(torch.load(latest_pth, map_location=DEVICE))
        start_from = latest_epoch
        logger.info(
            f"✅ 已加载第 {latest_epoch} 轮权重，将从第 {start_from + 1} 轮继续。"
        )
    else:
        logger.info("🆕 未发现历史权重，将从官方预训练大脑开始训练。")
        start_from = 0

    # 修正 4: 删除了之前画蛇添足的 model 重复定义，保留优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 5. 核心训练循环
    for epoch in range(start_from, EPOCHS):
        current_lr = INITIAL_LR * (0.5 ** (epoch // 3))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        logger.info(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr}")
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{EPOCHS}",
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

            if (i + 1) % 10 == 0:
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"}
                )

            if (i + 1) % 100 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        logger.info(f"===> Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")

        save_path = SAVE_DIR / f"checkpoint_epoch_letterbox_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"权重已保存: {save_path.name}")

    logger.info("--- 🚀 性能压榨版训练圆满完成！ ---")


if __name__ == "__main__":
    train()
