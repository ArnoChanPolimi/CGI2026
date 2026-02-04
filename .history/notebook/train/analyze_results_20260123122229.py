import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

# 导入你的模块
from dataset import PlantDataset, get_transforms
from model import get_plant_model

# 1. 路径配置
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train"
CSV_PATH = ROOT_DIR / "output" / "dataset_index" / "dataset_index.csv"
LOG_PATH = SAVE_DIR / "cpu_training_v1.log"
MODEL_PATH = SAVE_DIR / "checkpoint_epoch_5.pth"
DEVICE = torch.device("cpu")


def analyze():
    print("--- 🔍 开始全方位结果分析 ---")

    # ========================== 1. 绘制 Loss 曲线 ==========================
    print("正在从日志提取 Loss 数据...")
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 提取 Step Loss 和 Epoch Loss
    step_losses = [float(x) for x in re.findall(r"Loss: ([\d.]+)", content)]
    epoch_avg_losses = [float(x) for x in re.findall(r"平均 Loss: ([\d.]+)", content)]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(step_losses, color="#2ecc71", alpha=0.4, label="Batch Loss")
    plt.title("Training Loss (Step-by-Step)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(epoch_avg_losses) + 1),
        epoch_avg_losses,
        marker="s",
        color="#e74c3c",
        linewidth=2,
    )
    plt.title("Average Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, 6))
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "loss_trend.png")
    print(f"✅ Loss 趋势图已保存至: {SAVE_DIR / 'loss_trend.png'}")

    # ========================== 2. 运行测试集验证 ==========================
    print("\n正在加载模型并扫描测试集 (约 11GB 数据中的 test 部分)...")
    _, test_tf = get_transforms()
    test_ds = PlantDataset(CSV_PATH, ROOT_DIR, split="test", transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ========================== 3. 生成混淆矩阵 & 报告 ==========================
    # 计算准确率
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=["Abiotic", "Biotic"]
    )

    print("\n--- 分类报告 ---")
    print(report)

    # 绘制混淆矩阵图
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Abiotic", "Biotic"],
        yticklabels=["Abiotic", "Biotic"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.savefig(SAVE_DIR / "confusion_matrix.png")
    print(f"✅ 混淆矩阵图已保存至: {SAVE_DIR / 'confusion_matrix.png'}")

    plt.show()


if __name__ == "__main__":
    analyze()
