import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

# ========================== 0. 环境配置 ==========================
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from notebook.train.model import get_plant_model

# 自动定位输出和数据路径
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
PERSISTENT_CSV = SAVE_DIR / "final_inference_metrics_th_search.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================== 1. 推理与评估 ==========================


def run_optimized_inference():
    # 数据增强与加载
    test_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if not INFERENCE_PATCH_DIR.exists():
        print(f"❌ 路径不存在: {INFERENCE_PATCH_DIR}")
        return

    inference_ds = datasets.ImageFolder(
        root=str(INFERENCE_PATCH_DIR), transform=test_tf
    )
    data_loader = DataLoader(inference_ds, batch_size=128, shuffle=False, num_workers=4)
    class_names = inference_ds.classes  # 动态获取类别名称

    # 加载模型
    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(PTH_PATH, map_location=DEVICE))
    model.eval()

    # 第一步：提取所有叶片的原始信心分
    leaf_results = {}
    all_samples = inference_ds.samples
    sample_idx = 0

    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="⚡ 正在提取全局信心分"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            for i in range(len(probs)):
                img_path, _ = all_samples[sample_idx]
                # 统一 ID 命名规则
                base_name = "_".join(Path(img_path).stem.split("_")[:-2])

                if base_name not in leaf_results:
                    leaf_results[base_name] = {"probs": [], "actual": labels[i].item()}

                leaf_results[base_name]["probs"].append(probs[i])
                sample_idx += 1

    # 第二步：自动化阈值寻优
    print(f"\n🔍 正在基于 Epoch {BEST_EPOCH} 搜索最佳判定门槛...")

    records = []
    # 尝试多种门槛，看哪一个能救回 Abiotic
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.95]

    for th in thresholds:
        final_preds, final_labels = [], []

        for data in leaf_results.values():
            avg_probs = np.mean(data["probs"], axis=0)  # 计算该叶片 8 个切片的平均概率

            # 核心判定逻辑：只有 Biotic 的信心超过 th 才判为 1
            voted_res = 1 if avg_probs[1] > th else 0
            final_preds.append(voted_res)
            final_labels.append(data["actual"])

        # 计算指标
        cm = confusion_matrix(final_labels, final_preds)
        acc = 100.0 * (np.array(final_preds) == np.array(final_labels)).mean()

        # 统计混淆矩阵数据
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        print(
            f"门槛 {th:.2f} | Acc: {acc:.2f}% | Abiotic 正确: {tn}/476 | Biotic 正确: {tp}/4546"
        )

        records.append(
            {
                "epoch": BEST_EPOCH,
                "threshold": th,
                "acc": round(acc, 4),
                "abiotic_correct": tn,
                "biotic_correct": tp,
                "f1": round(f1_score(final_labels, final_preds, average="weighted"), 4),
            }
        )

        # 为每一个门槛生成对应的混淆矩阵图，方便对比
        save_cm_plot(final_labels, final_preds, class_names, th)

    # 保存 CSV
    df = pd.DataFrame(records)
    df.to_csv(PERSISTENT_CSV, mode="a", index=False, header=not PERSISTENT_CSV.exists())
    print(f"\n✅ 寻优完成，数据已记录至: {PERSISTENT_CSV.name}")


def save_cm_plot(labs, preds, class_names, th):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(labs, preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix (Threshold: {th})\nEpoch: {BEST_EPOCH}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(SAVE_DIR / f"cm_th_{th}_epoch_{BEST_EPOCH}.png")
    plt.close()


if __name__ == "__main__":
    run_optimized_inference()
