# notebook\train\patch_train_engine_v1.py
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

DATA_ROOT = ROOT_DIR / "data_processed" / "train"
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1" / "pth"
LOG_DIR = ROOT_DIR / "log" / "train" / "All_crop_train_logic_v1"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FILE_PREFIX = "patch_classifier"
logger = get_logger(str(LOG_DIR), "patch_training_v1.log")

BATCH_SIZE = 256
EPOCHS = 15
INITIAL_LR = 0.001
DEVICE = torch.device("cpu")
# ================================================


def get_transforms():
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
    # --- 📔 账本初始化逻辑 ---
    metrics_file = LOG_DIR / "train_metrics.csv"
    # 如果是重头开始训练，创建带表头的空文件；如果是接力，则不操作直接追加
    if not metrics_file.exists():
        with open(metrics_file, "w") as f:
            f.write("epoch,train_loss,train_acc\n")

    logger.info("--- 🚀 开启 Patch 级二分类引擎 v1 ---")

    train_ds = datasets.ImageFolder(root=str(DATA_ROOT), transform=get_transforms())

    # 类别权重计算
    abiotic_count = len(list((DATA_ROOT / "abiotic").glob("*")))
    biotic_count = len(list((DATA_ROOT / "biotic").glob("*")))
    weights = 1.0 / torch.tensor([abiotic_count, biotic_count], dtype=torch.float)
    class_weights = (weights / weights.sum() * 2.0).to(DEVICE)

    # train_loader = DataLoader(
    #     train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=12
    # )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,  # 匹配物理核心
        pin_memory=True,  # 内存锁定，加速读取
        persistent_workers=True,  # 换轮次不重开进程，省时间
        prefetch_factor=2,  # 预加载，不让 CPU 等数据
    )

    model = get_plant_model(num_classes=2).to(DEVICE)

    # 接力逻辑
    all_pths = list(SAVE_DIR.glob(f"{FILE_PREFIX}_epoch_*.pth"))
    start_epoch = 0
    if all_pths:
        epochs_found = [
            int(re.findall(r"epoch_(\d+).pth", p.name)[0]) for p in all_pths
        ]
        start_epoch = max(epochs_found)
        model.load_state_dict(
            torch.load(
                SAVE_DIR / f"{FILE_PREFIX}_epoch_{start_epoch}.pth", map_location=DEVICE
            )
        )
        logger.info(f"🔄 从第 {start_epoch+1} 轮继续...")

    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- 🔴 核心训练循环 ---
    for epoch in range(start_epoch, EPOCHS):
        current_lr = INITIAL_LR * (0.5 ** (epoch // 5))
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        model.train()
        running_loss = 0.0
        correct_total = 0
        samples_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 1. 累加 Loss
            running_loss += loss.item()
            # 2. 计算准确率 (Acc)
            _, predicted = torch.max(outputs.data, 1)
            samples_total += labels.size(0)
            correct_total += (predicted == labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.*correct_total/samples_total:.2f}%",
                }
            )

        # --- ✍️ 每一轮结束：计算平均值并存入 CSV ---
        final_avg_loss = running_loss / len(train_loader)
        final_acc = 100.0 * correct_total / samples_total

        with open(metrics_file, "a") as f:
            f.write(f"{epoch+1},{final_avg_loss:.4f},{final_acc:.2f}\n")

        logger.info(
            f"✅ Epoch {epoch+1} 结束 | Loss: {final_avg_loss:.4f} | Acc: {final_acc:.2f}%"
        )

        # 保存权重
        torch.save(model.state_dict(), SAVE_DIR / f"{FILE_PREFIX}_epoch_{epoch+1}.pth")

    logger.info("--- ✅ 训练完成，数据已全部存入 CSV 和 PTH ---")


if __name__ == "__main__":
    train()
