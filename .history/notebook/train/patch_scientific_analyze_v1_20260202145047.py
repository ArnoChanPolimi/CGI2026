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
# 使用 ImageFolder 基础类，但我们需要解析每个文件的路径
test_ds = datasets.ImageFolder(root=str(TEST_PATCH_DIR), transform=test_tf)


def evaluate_epoch_voted(pth_path, model):
    """
    【核心测试逻辑】：大图 ID 投票法
    1. 遍历 test 文件夹下所有 Patch
    2. 提取文件名中的 base_name (原图ID)
    3. 汇总同一 ID 的预测结果，少数服从多数判定 Biotic/Abiotic
    """
    model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
    model.eval()

    # voting_box 结构: { "叶子ID": {"preds": [0,1,1...], "actual": 1} }
    voting_box = {}

    print(f"🔬 正在对权重 {pth_path.name} 进行整图投票评估...")
    with torch.no_grad():
        for img_path, label in test_ds.samples:
            # 根据 process_data.py 的命名: f"{base_name}_tile_{count}.jpg"
            # 去掉末尾的 _tile_x.jpg 还原 base_name
            file_stem = Path(img_path).stem
            base_name = "_".join(file_stem.split("_")[:-2])

            # 加载并预测单张 Patch
            img = (
                test_tf(datasets.folder.default_loader(img_path))
                .unsqueeze(0)
                .to(DEVICE)
            )
            output = model(img)
            pred = output.argmax(1).item()

            if base_name not in voting_box:
                voting_box[base_name] = {"preds": [], "actual": label}
            voting_box[base_name]["preds"].append(pred)

    # 执行投票判定
    final_preds, final_labels = [], []
    for leaf_id, data in voting_box.items():
        votes = data["preds"]
        # 2分类投票：1的占比超过 0.5 则判定为 1 (Biotic)
        voted_res = 1 if (sum(votes) / len(votes)) > 0.5 else 0
        final_preds.append(voted_res)
        final_labels.append(data["actual"])

    # 计算基于“叶子张数”的科研指标
    metrics = {
        "test_acc": 100.0 * (np.array(final_preds) == np.array(final_labels)).mean(),
        "test_f1": f1_score(final_labels, final_preds, average="weighted"),
        "test_recall": recall_score(final_labels, final_preds, average="weighted"),
        "test_precision": precision_score(
            final_labels, final_preds, average="weighted"
        ),
        "test_loss": 0.0,  # 投票法通常不计算单片 Loss 的平均，科研中以 Acc/F1 为准
    }
    return metrics, final_labels, final_preds


def sync_and_analyze():
    # A. 初始化
    cols = [
        "epoch",
        "train_loss",
        "train_acc",
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
    df_train = pd.read_csv(TRAIN_LOG_CSV)
    model = get_plant_model(num_classes=2).to(DEVICE)
    pth_files = sorted(
        list(PTH_DIR.glob("patch_classifier_epoch_*.pth")),
        key=lambda x: int(re.findall(r"epoch_(\d+)", x.name)[0]),
    )

    # B. 增量计算 (基于整图投票)
    for pth in pth_files:
        epoch = int(re.findall(r"epoch_(\d+)", pth.name)[0])
        if epoch in df_p["epoch"].values:
            continue

        m, _, _ = evaluate_epoch_voted(pth, model)
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

    df_p = df_p.sort_values("epoch")
    df_p.to_csv(PERSISTENT_CSV, index=False)

    # C. 趋势大图
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(df_p["epoch"], df_p["train_acc"], "r-o", label="Train Acc (Patch)")
    plt.plot(df_p["epoch"], df_p["test_acc"], "b-s", label="Test Acc (Leaf-Voted)")
    plt.title("Accuracy Comparison (Patch vs Voted-Leaf)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(df_p["epoch"], df_p["test_f1"], "g-^", label="Leaf F1-Score")
    plt.plot(df_p["epoch"], df_p["test_precision"], "m-v", label="Leaf Precision")
    plt.title("Leaf-Level Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(SAVE_DIR / "scientific_full_metrics.png")
    print(f"📊 趋势图已保存：scientific_full_metrics.png")


def plot_final_confusion_matrix(epoch_num=9):
    """针对指定轮次生成整张叶子的混淆矩阵"""
    model = get_plant_model(num_classes=2).to(DEVICE)
    target_pth = next(
        p for p in PTH_DIR.glob(f"patch_classifier_epoch_{epoch_num}.pth")
    )

    _, labs, preds = evaluate_epoch_voted(target_pth, model)

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
    plt.title(f"Leaf-Level Confusion Matrix (Epoch {epoch_num})")
    plt.ylabel("Actual Leaf Category")
    plt.xlabel("Voted Predicted Category")
    plt.savefig(SAVE_DIR / f"confusion_matrix_leaf_epoch_{epoch_num}.png")
    plt.close()
    print(f"✅ 混淆矩阵已保存：confusion_matrix_leaf_epoch_{epoch_num}.png")


if __name__ == "__main__":
    sync_and_analyze()
    plot_final_confusion_matrix(epoch_num=9)  # 这里指定你表现最好的那一轮
