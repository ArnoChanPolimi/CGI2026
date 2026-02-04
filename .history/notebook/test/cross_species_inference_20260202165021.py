import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from tqdm import tqdm

# ========================== 0. 路径与环境配置 ==========================
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from notebook.train.model import get_plant_model

SAVE_DIR = (
    PROJECT_ROOT
    / "output"
    / "All_crop_train_logic_v1"
    / "cross_species_test"
    / "test_inference"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BEST_EPOCH = 9
PTH_PATH = (
    PROJECT_ROOT
    / "output"
    / "All_crop_train_logic_v1"
    / "pth"
    / f"patch_classifier_epoch_{BEST_EPOCH}.pth"
)
INFERENCE_PATCH_DIR = PROJECT_ROOT / "Inference_Data_Processed"
PERSISTENT_CSV = SAVE_DIR / "final_inference_metrics.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================== 1. 数据加载 ==========================
test_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

inference_ds = datasets.ImageFolder(root=str(INFERENCE_PATCH_DIR), transform=test_tf)
data_loader = DataLoader(
    inference_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
)

# ========================== 2. 核心功能 ==========================


def run_final_inference():
    print(f"⚡ 启动实战评估 | 模式: 信心加权投票 | 目标: Epoch {BEST_EPOCH}")

    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(PTH_PATH, map_location=DEVICE))
    model.eval()

    voting_box = {}
    all_samples = inference_ds.samples
    sample_idx = 0

    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Soft Voting"):
            imgs = imgs.to(DEVICE)
            # 获取模型原始输出并转为概率 (Softmax)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            for i in range(len(probs)):
                img_path, _ = all_samples[sample_idx]
                base_name = "_".join(Path(img_path).stem.split("_")[:-2])

                if base_name not in voting_box:
                    # 存储 [Abiotic累加信心, Biotic累加信心], 真实标签
                    voting_box[base_name] = {
                        "conf_sums": np.array([0.0, 0.0]),
                        "actual": labels[i].item(),
                    }

                voting_box[base_name]["conf_sums"] += probs[i]
                sample_idx += 1

    # 执行决策：取累加信心最高的类别
    final_preds, final_labels = [], []
    for data in voting_box.values():
        voted_res = np.argmax(data["conf_sums"])
        final_preds.append(voted_res)
        final_labels.append(data["actual"])

    # 计算指标
    acc = 100.0 * (np.array(final_preds) == np.array(final_labels)).mean()
    f1 = f1_score(final_labels, final_preds, average="weighted")
    rec = recall_score(final_labels, final_preds, average="weighted")
    pre = precision_score(final_labels, final_preds, average="weighted")

    # --- 核心修改：增量保存 CSV ---
    new_data = {
        "epoch": BEST_EPOCH,
        "total_leaves": len(final_preds),
        "acc": acc,
        "f1": f1,
        "recall": rec,
        "precision": pre,
        "mode": "soft_voting_confidence",  # 标记一下这是信心加权模式
    }
    res_df = pd.DataFrame([new_data])

    # 检查文件是否存在
    file_exists = os.path.isfile(PERSISTENT_CSV)

    # mode='a' 表示 append (追加)
    # header=not file_exists 表示只有文件不存在时才写表头
    res_df.to_csv(PERSISTENT_CSV, mode="a", index=False, header=not file_exists)

    print(f"\n✅ 结果已追加至: {PERSISTENT_CSV}")
    print(f"📊 当前 Acc: {acc:.2f}% | F1: {f1:.4f}")

    # 保存混淆矩阵图 (会覆盖同名旧图，因为这是视觉结论)
    save_confusion_matrix(final_labels, final_preds)


def save_confusion_matrix(labs, preds):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(labs, preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=inference_ds.classes,
        yticklabels=inference_ds.classes,
    )
    plt.title(f"Final CM (Epoch {BEST_EPOCH}) - Soft Voting")
    plt.savefig(SAVE_DIR / f"final_cm_epoch_{BEST_EPOCH}_soft.png")
    plt.close()


if __name__ == "__main__":
    run_fast_inference()  # 这里的逻辑已经更新
    run_final_inference()
