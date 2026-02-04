import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

# 导入你的日志工具
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logger_utils import get_logger

log = get_logger()
os.makedirs("../output", exist_ok=True)

# --- 1. 数据加载逻辑 (略，保持之前的相对路径) ---
# (此处省略之前的 data_transforms 和 dataloaders 代码，确保它们在脚本中)

# --- 2. 增加：记录训练过程的容器 ---
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}


# --- 3. 训练核心逻辑 (带记录功能) ---
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward(), optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())
            log.info(
                f"Epoch {epoch}/{num_epochs-1} | {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
            )


# --- 4. 运行训练 ---
train_model(model, criterion, optimizer, num_epochs=10)

# --- 5. 【核心产出 A】画图并保存 ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig("../output/learning_curves.png")

# --- 6. 【核心产出 B & C】混淆矩阵与报告 ---
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in dataloaders["val"]:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

# 保存文字版报告
report = classification_report(y_true, y_pred, target_names=class_names)
with open("../output/classification_report.txt", "w") as f:
    f.write(report)
log.info("所有核心产出（曲线图、报告、模型）已存入 output 文件夹！")
