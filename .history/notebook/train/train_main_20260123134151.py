# notebook/train/train_main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import sys

from dataset import PlantDataset, get_transforms
from model import get_plant_model

# 处理路径和日志工具
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))
from notebook.tools.logger_utils import get_logger

# ================= 针对 11GB 数据 & CPU 的硬核配置 =================
CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 调用你的 logger 脚本，存放在 output/All_crop_train，文件名自定义
logger = get_logger(log_dir=str(SAVE_DIR), log_filename="cpu_training_v1.log")

BATCH_SIZE = 32  # 利用 32G 内存，提高 CPU 利用率
EPOCHS = 7  # 11GB 数据量大，先跑 5 轮查看效果
# LR = 0.0005  # 较小的学习率保证在不平衡数据下的稳定性
START_EPOCH = 5  # 告诉程序：我已经练过 5 轮了，从第 6 轮开始
LR = 0.00025  # 调小学习率，精修细节

DEVICE = torch.device("cpu")
# =================================================================


def train():
    logger.info("--- 开始训练任务 (针对 11GB 数据的优化版) ---")

    # 1. 计算权重，对抗“数据量不平衡”
    df = pd.read_csv(CSV_PATH)
    train_df = df[df["split"] == "train"]
    counts = train_df["label"].value_counts().sort_index().values

    # 逻辑：样本越少，权重越高。防止模型只学 AloeVera。
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    weights = weights / weights.sum() * 2.0  # 归一化并平衡
    class_weights = weights.to(DEVICE)

    logger.info(f"类别统计: Abiotic={counts[0]}, Biotic={counts[1]}")
    logger.info(f"损失函数补偿权重: {class_weights}")

    # 2. 准备数据加载
    train_tf, test_tf = get_transforms()
    train_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="train", transform=train_tf)
    # Windows CPU 建议 num_workers 为 0 或较小值，防止死锁
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    # 3. 初始化
    model = get_plant_model(num_classes=2).to(DEVICE)
    # 【新增：权重扫描与加载逻辑】
    checkpoint_path = SAVE_DIR / "checkpoint_epoch_5.pth"  # 假设你想加载第5轮的结果

    if checkpoint_path.exists():
        logger.info(f"🔍 发现已存在的权重文件: {checkpoint_path}，正在加载...")
        # map_location='cpu' 确保在没 GPU 的电脑上也能顺利读取
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        logger.info("✅ 权重加载成功！将在现有基础上继续训练/测试。")
    else:
        logger.info("🆕 未发现预存权重，将从官方预训练大脑开始训练。")

    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 应用补偿权重
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 【新增逻辑】：学习率调度器
    # 每隔 3 个 Epoch，学习率乘以 0.5
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # 4. 核心训练循环
    # for epoch in range(EPOCHS):
    # 改成这一行
    for epoch in range(START_EPOCH, EPOCHS):
        # --- 硬核手动控制：第6轮用0.00025，第7轮自动减半成0.000125 ---
        lr_to_use = 0.00025 if epoch == 5 else 0.000125
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_to_use
        # ----------------------------------------------------------

        model.train()
        running_loss = 0.0

        logger.info(f"开始第 {epoch+1}/{EPOCHS} 轮训练...")

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 标准三步走：清零梯度 -> 前向计算 -> 反向传播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 进度反馈：每 100 步在屏幕闪烁一次，每 500 步记入日志文件
            if (i + 1) % 100 == 0:
                msg = f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                print(msg)
                if (i + 1) % 500 == 0:
                    logger.info(msg)

        # 每个 Epoch 结束后的收尾动作
        avg_loss = running_loss / len(train_loader)

        # 更新学习率并记录
        # scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # logger.info(
        #     f"===> Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}, 下轮学习率: {current_lr}"
        # )
        # 修改日志打印，直接用刚才定义的 lr_to_use
        logger.info(f"===> Epoch {epoch+1} 完成，本轮 LR: {lr_to_use}")
        logger.info(f"平均 Loss: {avg_loss:.4f}")

        # 实时保存，防止 Windows 意外自动重启或断电
        save_path = SAVE_DIR / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"模型权重已保存至: {save_path.name}")

    logger.info("--- 7 轮训练任务圆满完成！ ---")
    logger.info("--- 训练结束，准备后续分析 ---")


if __name__ == "__main__":
    train()
