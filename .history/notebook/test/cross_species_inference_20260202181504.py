import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from tqdm import tqdm

# ========================== 0. 环境配置 ==========================
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

BEST_EPOCH = 14
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

# ========================== 1. 核心推理 ==========================


def run_optimized_inference():
    test_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    inference_ds = datasets.ImageFolder(
        root=str(INFERENCE_PATCH_DIR), transform=test_tf
    )
    data_loader = DataLoader(inference_ds, batch_size=128, shuffle=False, num_workers=4)

    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(PTH_PATH, map_location=DEVICE))
    model.eval()

    # 存储池：记录每个叶片的所有切片概率
    # { "base_name": {"probs": [[p0, p1], [p0, p1]...], "actual": label} }
    leaf_results = {}
    all_samples = inference_ds.samples
    sample_idx = 0

    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="⚡ 正在提取全局信心特征"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            for i in range(len(probs)):
                img_path, _ = all_samples[sample_idx]
                base_name = "_".join(Path(img_path).stem.split("_")[:-2])

                if base_name not in leaf_results:
                    leaf_results[base_name] = {"probs": [], "actual": labels[i].item()}

                leaf_results[base_name]["probs"].append(probs[i])
                sample_idx += 1

    # ========================== 2. 自动化阈值寻优 ==========================
    print("\n🔍 正在寻找拯救 Abiotic 的黄金门槛...")

    results_to_save = []
    # 尝试从 0.5 到 0.95 的门槛
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        final_preds, final_labels = [], []

        for base_name, data in leaf_results.items():
            # 计算平均概率
            avg_probs = np.mean(data["probs"], axis=0)

            # 门槛逻辑：只有 Biotic (index 1) 信心超过 threshold 才判为 1
            voted_res = 1 if avg_probs[1] > threshold else 0

            final_preds.append(voted_res)
            final_labels.append(data["actual"])

        # 计算指标
        acc = 100.0 * (np.array(final_preds) == np.array(final_labels)).mean()
        cm = confusion_matrix(final_labels, final_preds)

        # 记录识别情况
        abiotic_correct = cm[0, 0] if cm.shape[0] > 0 else 0
        biotic_correct = cm[1, 1] if cm.shape[1] > 1 else 0

        print(
            f"门槛 {threshold:.2f} -> Acc: {acc:.2f}% | 救回 Abiotic: {abiotic_correct}/476 | 牺牲 Biotic: {4546-biotic_correct}"
        )

        # 准备存入 CSV 的数据
        results_to_save.append(
            {
                "epoch": BEST_EPOCH,
                "threshold": threshold,
                "acc": round(acc, 4),
                "abiotic_hit": abiotic_correct,
                "biotic_hit": biotic_correct,
                "f1": round(f1_score(final_labels, final_preds, average="weighted"), 4),
                "method": f"soft_voting_th_{threshold}",
            }
        )

        # 顺手保存一张你最想看的 0.9 混淆矩阵
        if threshold == 0.9:
            save_cm(final_labels, final_preds, threshold)

    # 增量保存
    df_save = pd.DataFrame(results_to_save)
    file_exists = PERSISTENT_CSV.exists()
    df_save.to_csv(PERSISTENT_CSV, mode="a", index=False, header=not file_exists)
    print(f"\n✅ 所有阈值测试结果已追加至: {PERSISTENT_CSV.name}")


def save_cm(labs, preds, th):
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
    plt.title(f"Confusion Matrix (Threshold: {th})\nAbiotic Focus Mode")
    plt.savefig(SAVE_DIR / f"cm_epoch_{BEST_EPOCH}_th_{th}.png")
    plt.close()


if __name__ == "__main__":
    run_optimized_inference()
