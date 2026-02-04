# notebook\train\train_main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# 导入你写好的工具
from dataset import PlantDataset, get_transforms
from model import get_plant_model
import sys

# 确保能找到 tools 文件夹
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tools.logger_utils import get_logger

# ================= 配置区 (全部相对路径) =================
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent

# 输入：账本路径
CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"

# 输出：训练成果路径
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 日志配置
logger = get_logger(log_dir=str(SAVE_DIR), log_filename="training_process.log")

# 超参数
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ========================================================


def train():
    logger.info(f"--- 启动全作物训练任务 | 设备: {DEVICE} ---")

    # 1. 准备数据
    train_tf, test_tf = get_transforms()
    train_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="train", transform=train_tf)
    test_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="test", transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 准备模型
    model = get_plant_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 用于记录绘图数据
    history = {"train_loss": [], "test_acc": []}
    best_acc = 0.0

    # 3. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证环节
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 收集混淆矩阵数据
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history["train_loss"].append(epoch_loss)
        history["test_acc"].append(epoch_acc)

        logger.info(
            f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Test Acc: {epoch_acc:.2f}%"
        )

        # 4. 保存最强模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), SAVE_DIR / "best_model.pth")
            logger.info(f"✨ 检测到更高准确率，模型已更新保存。")

    # 5. 绘制曲线图与混淆矩阵
    draw_results(history, all_labels, all_preds, SAVE_DIR)
    logger.info("--- 训练结束，所有成果已存入 output/All_crop_train ---")


def draw_results(history, labels, preds, save_path):
    # 损失与准确率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["test_acc"], label="Test Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig(save_path / "learning_curves.png")

    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Abiotic", "Biotic"]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(save_path / "confusion_matrix.png")


if __name__ == "__main__":
    train()
