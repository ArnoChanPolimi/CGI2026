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

# ========================== 0. 路径与环境配置 ==========================
current_file = Path(__file__).resolve()
# 自动回溯到根目录 CGI_PROJECT
PROJECT_ROOT = current_file.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from notebook.train.model import get_plant_model

# 指定输出位置：隔离旧逻辑
SAVE_DIR = (
    PROJECT_ROOT
    / "output"
    / "All_crop_train_logic_v1"
    / "cross_species_test"
    / "test_inference"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 核心输入配置
BEST_EPOCH = 14  # 9
PTH_PATH = (
    PROJECT_ROOT
    / "output"
    / "All_crop_train_logic_v1"
    / "pth"
    / f"patch_classifier_epoch_{BEST_EPOCH}.pth"
)
INFERENCE_PATCH_DIR = PROJECT_ROOT / "Inference_Data_Processed"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================== 1. 数据加载优化 ==========================
test_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

if not INFERENCE_PATCH_DIR.exists():
    print(f"❌ 错误：找不到切片文件夹 {INFERENCE_PATCH_DIR}")
    sys.exit()

inference_ds = datasets.ImageFolder(root=str(INFERENCE_PATCH_DIR), transform=test_tf)

# 使用 DataLoader 开启并行读图模式
# Windows 建议 num_workers 设为 4，Linux 可设为 8 或更高
data_loader = DataLoader(
    inference_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
)

# ========================== 2. 核心功能与逻辑 ==========================


def save_confusion_matrix(labs, preds):
    """生成并保存混淆矩阵图"""
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
    plt.title(
        f"Final Real-World Confusion Matrix (Epoch {BEST_EPOCH})\n[Unit: Single Leaf]"
    )
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label (Voted)")
    plt.savefig(SAVE_DIR / f"final_cm_epoch_{BEST_EPOCH}.png")
    plt.close()
    print(f"📊 混淆矩阵已保存至: {SAVE_DIR / f'final_cm_epoch_{BEST_EPOCH}.png'}")


def run_fast_inference():
    """
    逻辑：
    1. 加载最佳 PTH 权重。
    2. 批量推理切片，通过 Index 关联原始文件名还原叶片 ID。
    3. 执行多数投票制（Voted）决策。
    4. 输出 CSV 成绩单和混淆矩阵。
    """
    print(f"⚡ 启动闪电推理 | 目标: Epoch {BEST_EPOCH} | 设备: {DEVICE}")

    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(PTH_PATH, map_location=DEVICE))
    model.eval()

    voting_box = {}  # 格式: { "叶片ID": {"preds": [], "actual": label} }
    all_samples = inference_ds.samples
    sample_idx = 0

    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Parallel Inferencing"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()

            # 批量解析预测结果
            for i in range(len(preds)):
                img_path, _ = all_samples[sample_idx]
                # 还原 base_name (品种_病害_原文件名)
                file_stem = Path(img_path).stem
                base_name = "_".join(file_stem.split("_")[:-2])

                if base_name not in voting_box:
                    voting_box[base_name] = {"preds": [], "actual": labels[i].item()}

                voting_box[base_name]["preds"].append(preds[i])
                sample_idx += 1

    # 执行单张叶片级别的投票决策
    final_preds, final_labels = [], []
    for leaf_id, data in voting_box.items():
        votes = data["preds"]
        voted_res = 1 if (sum(votes) / len(votes)) > 0.5 else 0
        final_preds.append(voted_res)
        final_labels.append(data["actual"])

    # 计算各项关键指标
    acc = 100.0 * (np.array(final_preds) == np.array(final_labels)).mean()
    f1 = f1_score(final_labels, final_preds, average="weighted")
    rec = recall_score(final_labels, final_preds, average="weighted")
    pre = precision_score(final_labels, final_preds, average="weighted")

    # 打印最终报告
    print("\n" + "★" * 30)
    print(f"🏆 实战评估报告 (Epoch {BEST_EPOCH})")
    print(f"📸 评估叶片总数: {len(final_preds)}")
    print(f"🎯 准确率 (Acc): {acc:.2f}%")
    print(f"🎭 F1 Score: {f1:.4f}")
    print(f"🔍 召回率 (Recall): {rec:.4f}")
    print(f"✨ 精确率 (Precision): {pre:.4f}")
    print("★" * 30)

    # 1. 输出 CSV 指标
    res_df = pd.DataFrame(
        [
            {
                "epoch": BEST_EPOCH,
                "total_leaves": len(final_preds),
                "acc": acc,
                "f1": f1,
                "recall": rec,
                "precision": pre,
            }
        ]
    )
    res_df.to_csv(SAVE_DIR / "final_inference_metrics.csv", index=False)

    # 2. 输出混淆矩阵图
    save_confusion_matrix(final_labels, final_preds)


if __name__ == "__main__":
    run_fast_inference()
