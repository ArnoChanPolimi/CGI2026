# notebook\train\patch_scientific_analyze_v1.py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

# ========================== 0. 路径修正 ==========================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from notebook.train.model import get_plant_model

# ========================== 1. 路径配置 ==========================
ROOT_DIR = project_root
TRAIN_LOG_CSV = (
    ROOT_DIR
    / "log"
    / "train"
    / "All_crop_train_logic_v1"
    / "train_metrics_patch_v1.csv"
)
SAVE_DIR = ROOT_DIR / "log" / "analyze" / "patch_v1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PERSISTENT_CSV = SAVE_DIR / "persistent_metrics.csv"
PTH_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1" / "pth"
TEST_PATCH_DIR = ROOT_DIR / "data_processed" / "test"

DEVICE = torch.device("cpu")

# ========================== 2. 数据准备 ==========================
test_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_ds = datasets.ImageFolder(root=str(TEST_PATCH_DIR), transform=test_tf)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)


def evaluate_epoch(pth_path, model):
    """阅卷：算出当前 Epoch 的所有科研指标"""
    model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    total_loss, criterion = 0.0, torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, lbls in test_loader:
            outputs = model(imgs.to(DEVICE))
            total_loss += criterion(outputs, lbls.to(DEVICE)).item()
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    # 计算该轮次的全部指标
    metrics = {
        "test_loss": total_loss / len(test_loader),
        "test_acc": 100.0 * (np.array(all_preds) == np.array(all_labels)).mean(),
        "test_f1": f1_score(all_labels, all_preds, average="weighted"),
        "test_recall": recall_score(all_labels, all_preds, average="weighted"),
        "test_precision": precision_score(all_labels, all_preds, average="weighted"),
    }
    return metrics, all_labels, all_preds


def sync_and_analyze():
    # A. 初始化数据表
    cols = [
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "test_f1",
        "test_recall",
        "test_precision",
    ]
    df_p = (
        pd.read_csv(PERSISTENT_CSV)
        if PERSISTENT_CSV.exists()
        else pd.DataFrame(columns=cols)
    )

    # B. 读取原始训练日志
    df_train = pd.read_csv(TRAIN_LOG_CSV)
    model = get_plant_model(num_classes=2).to(DEVICE)

    # C. 增量计算
    pth_files = sorted(
        list(PTH_DIR.glob("patch_classifier_epoch_*.pth")),
        key=lambda x: int(re.findall(r"epoch_(\d+)", x.name)[0]),
    )

    for pth in pth_files:
        epoch = int(re.findall(r"epoch_(\d+)", pth.name)[0])
        if epoch in df_p["epoch"].values:
            continue

        print(f"📊 正在计算第 {epoch} 轮全指标...")
        m, labs, preds = evaluate_epoch(pth, model)

        train_row = df_train[df_train["epoch"] == epoch]
        if not train_row.empty:
            m.update(
                {
                    "epoch": epoch,
                    "train_loss": train_row["train_loss"].values[0],
                    "train_acc": train_row["train_acc"].values[0],
                }
            )
            df_p = pd.concat([df_p, pd.DataFrame([m])], ignore_index=True)

    # D. 保存数据并画大图
    df_p = df_p.sort_values("epoch")
    df_p.to_csv(PERSISTENT_CSV, index=False)

    # E. 科学画图：4子图模式
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # 1. Loss
    axes[0, 0].plot(df_p["epoch"], df_p["train_loss"], "r-o", label="Train")
    axes[0, 0].plot(df_p["epoch"], df_p["test_loss"], "b-s", label="Test")
    axes[0, 0].set_title("Loss Trend")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    # 2. Accuracy
    axes[0, 1].plot(df_p["epoch"], df_p["train_acc"], "r-o", label="Train")
    axes[0, 1].plot(df_p["epoch"], df_p["test_acc"], "b-s", label="Test")
    axes[0, 1].set_title("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    # 3. F1-Score
    axes[1, 0].plot(df_p["epoch"], df_p["test_f1"], "g-^", label="Test F1")
    axes[1, 0].set_title("F1-Score Trend")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    # 4. Precision & Recall
    axes[1, 1].plot(df_p["epoch"], df_p["test_precision"], "m-v", label="Precision")
    axes[1, 1].plot(df_p["epoch"], df_p["test_recall"], "y-x", label="Recall")
    axes[1, 1].set_title("Precision & Recall")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "scientific_full_metrics.png")
    print(
        f"🏁 分析完成！\n1. 数据表：{PERSISTENT_CSV}\n2. 趋势大图：scientific_full_metrics.png"
    )


def plot_specific_confusion_matrix(epoch_num):
    """
    单独画混淆矩阵的功能：
    1. 自动找到对应 epoch 的 pth 文件
    2. 加载模型并在测试集上跑一遍
    3. 生成混淆矩阵图片
    """
    print(f"🎯 正在为第 {epoch_num} 轮生成专属混淆矩阵...")

    # A. 寻找对应的 pth 文件
    pth_files = list(PTH_DIR.glob(f"patch_classifier_epoch_{epoch_num}.pth"))
    if not pth_files:
        print(f"❌ 错误：在路径下没找到第 {epoch_num} 轮的 pth 文件！")
        return

    # B. 加载模型并评估
    model = get_plant_model(num_classes=2).to(DEVICE)
    _, labs, preds = evaluate_epoch(pth_files[0], model)

    # C. 绘图并保存
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(labs, preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=test_ds.classes,
        yticklabels=test_ds.classes,
    )
    plt.title(f"Confusion Matrix - Epoch {epoch_num}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    save_path = SAVE_DIR / f"confusion_matrix_epoch_{epoch_num}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 混淆矩阵已成功输出至: {save_path}")


if __name__ == "__main__":
    sync_and_analyze()
