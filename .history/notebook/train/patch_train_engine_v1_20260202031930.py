import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import sys
import os
import re
from tqdm import tqdm

# ================= 🚀 核心配置区 =================
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from notebook.tools.logger_utils import get_logger
from notebook.train.model import get_plant_model

# 路径配置
DATA_ROOT = ROOT_DIR / "data_processed" / "train"
# 按你要求的逻辑命名存储路径
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1" / "pth"
LOG_DIR = ROOT_DIR / "log" / "train" / "All_crop_train_logic_v1"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 权重文件前缀
FILE_PREFIX = "patch_classifier"
logger = get_logger(str(LOG_DIR), "patch_training_v1.log")

# 压榨性能配置 (i5-13500H)
BATCH_SIZE = 128
EPOCHS = 15
INITIAL_LR = 0.001
DEVICE = torch.device("cpu")  # 强制 CPU 模式
# ================================================


def get_transforms():
    """功能：针对切片的图像增强，增加模型泛化力"""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def train():
    logger.info("======================================================")
    logger.info(f"--- 🚀 开启 Patch 级二分类引擎 v1 ---")

    # 1. 加载数据 (ImageFolder 自动处理类别)
    train_ds = datasets.ImageFolder(root=str(DATA_ROOT), transform=get_transforms())

    # 自动计算类别权重 (根据文件夹内图片数量)
    # class_to_idx 通常是 {'abiotic': 0, 'biotic': 1}
    abiotic_count = len(os.listdir(DATA_ROOT / "abiotic"))
    biotic_count = len(os.listdir(DATA_ROOT / "biotic"))
    counts = torch.tensor([abiotic_count, biotic_count], dtype=torch.float)
    weights = 1.0 / counts
    class_weights = (weights / weights.sum() * 2.0).to(DEVICE)

    logger.info(f"📊 样本分布: Abiotic={abiotic_count}, Biotic={biotic_count}")
    logger.info(f"⚖️ 类别权重: {class_weights.tolist()}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=False
    )

    # 2. 初始化模型
    model = get_plant_model(num_classes=2).to(DEVICE)

    # 3. 扫描并接力历史权重
    all_pths = list(SAVE_DIR.glob(f"{FILE_PREFIX}_epoch_*.pth"))
    start_epoch = 0
    if all_pths:
        try:
            epochs = [
                int(re.findall(f"{FILE_PREFIX}_epoch_(\d+).pth", p.name)[0])
                for p in all_pths
            ]
            start_epoch = max(epochs)
            latest_pth = SAVE_DIR / f"{FILE_PREFIX}_epoch_{start_epoch}.pth"
            model.load_state_dict(torch.load(latest_pth, map_location=DEVICE))
            logger.info(
                f"🔄 发现旧大脑：{latest_pth.name}，从第 {start_epoch+1} 轮继续..."
            )
        except Exception as e:
            logger.warning(f"权重加载失败，重头开始: {e}")

    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. 训练主循环
    for epoch in range(start_epoch, EPOCHS):
        # 简单学习率调度：每 5 轮减半
        current_lr = INITIAL_LR * (0.5 ** (epoch // 5))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})

        avg_loss = running_loss / len(train_loader)
        logger.info(f"===> Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")

        # 认认真真保存
        save_path = SAVE_DIR / f"{FILE_PREFIX}_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"💾 权重已保存: {save_path.name}")

    logger.info("--- ✅ 训练任务圆满完成！ ---")


if __name__ == "__main__":
    train()
