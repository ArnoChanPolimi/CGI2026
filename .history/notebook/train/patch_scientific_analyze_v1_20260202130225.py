import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ========================== 1. 路径配置 ==========================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# 我们自己维护的持久化日志，专门存 4 条线的数据
PERSISTENT_CSV = ROOT_DIR / "log" / "analyze" / "patch_v1" / "persistent_metrics.csv"
TRAIN_LOG_CSV = (
    ROOT_DIR / "log" / "train" / "All_crop_train_logic_v1" / "train_metrics.csv"
)
PTH_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1" / "pth"
TEST_PATCH_DIR = ROOT_DIR / "data_processed" / "test"
SAVE_DIR = ROOT_DIR / "log" / "analyze" / "patch_v1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

from notebook.train.model import get_plant_model

DEVICE = torch.device("cpu")  # 阅卷通常用 CPU 即可，不占训练显存

test_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_test_metrics(pth_path):
    """单独测算某一个权重在测试集上的平均 Loss 和 Acc"""
    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
    model.eval()

    from torch.utils.data import DataLoader, Dataset

    # 这里复用之前的切片读取逻辑，或者简单的 ImageFolder
    # 为保证效率，建议使用 PyTorch 的 DataLoader
    from torchvision.datasets import ImageFolder

    test_ds = ImageFolder(root=str(TEST_PATCH_DIR), transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, lbls in test_loader:
            outputs = model(imgs.to(DEVICE))
            loss = criterion(outputs, lbls.to(DEVICE))
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += lbls.size(0)
            correct += (predicted == lbls.to(DEVICE)).sum().item()

    return total_loss / len(test_loader), 100.0 * correct / total


def sync_and_analyze():
    # --- A. 初始化或加载持久化 CSV ---
    if PERSISTENT_CSV.exists():
        df_p = pd.read_csv(PERSISTENT_CSV)
        recorded_epochs = df_p["epoch"].tolist()
    else:
        df_p = pd.DataFrame(
            columns=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"]
        )
        recorded_epochs = []

    # --- B. 获取训练原始数据 (用于同步 Train Loss/Acc) ---
    if not TRAIN_LOG_CSV.exists():
        print(f"❌ 未找到训练日志 {TRAIN_LOG_CSV}")
        return
    df_train = pd.read_csv(TRAIN_LOG_CSV)

    # --- C. 扫描 PTH 文件夹，寻找待增量更新的轮次 ---
    pth_files = list(PTH_DIR.glob("patch_classifier_epoch_*.pth"))
    new_records = []

    for pth in pth_files:
        epoch = int(re.findall(r"epoch_(\d+)", pth.name)[0])

        # 如果这一轮还没被记录在持久化 CSV 里
        if epoch not in recorded_epochs:
            print(f"🔎 发现新轮次 Epoch {epoch}，正在补算测试集指标...")

            # 获取对应的训练指标
            train_row = df_train[df_train["epoch"] == epoch]
            if train_row.empty:
                continue  # 训练还没写完这一轮，跳过

            t_loss, t_acc = get_test_metrics(pth)

            new_records.append(
                {
                    "epoch": epoch,
                    "train_loss": train_row["train_loss"].values[0],
                    "train_acc": train_row["train_acc"].values[0],
                    "test_loss": t_loss,
                    "test_acc": t_acc,
                }
            )

    # --- D. 合并并保存 ---
    if new_records:
        df_new = pd.DataFrame(new_records)
        df_p = pd.concat([df_p, df_new], ignore_index=True).sort_values(by="epoch")
        df_p.to_csv(PERSISTENT_CSV, index=False)
        print(f"✅ 持久化 CSV 已更新，当前共记录 {len(df_p)} 轮。")

    # --- E. 基于持久化 CSV 画图 (4条线) ---
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df_p["epoch"], df_p["train_loss"], "r-o", label="Train Loss")
    plt.plot(df_p["epoch"], df_p["test_loss"], "b-s", label="Test Loss")
    plt.title("Loss Curve (Persistent)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df_p["epoch"], df_p["train_acc"], "r-o", label="Train Acc")
    plt.plot(df_p["epoch"], df_p["test_acc"], "b-s", label="Test Acc")
    plt.title("Accuracy Curve (Persistent)")
    plt.legend()
    plt.grid(True)

    plt.savefig(SAVE_DIR / "persistent_4lines_plot.png")
    print(f"📈 4线图已刷新: {SAVE_DIR / 'persistent_4lines_plot.png'}")


if __name__ == "__main__":
    sync_and_analyze()
