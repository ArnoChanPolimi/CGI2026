import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import re
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from tqdm import tqdm

# ========================== 0. 路径与环境配置 ==========================
current_file = Path(__file__).resolve()
# 确保 PROJECT_ROOT 指向项目根目录 CGI_PROJECT
PROJECT_ROOT = current_file.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 导入你的模型定义
from notebook.train.model import get_plant_model

# 指定输出位置：隔离旧逻辑，专门存放实战结果
SAVE_DIR = (
    PROJECT_ROOT
    / "output"
    / "All_crop_train_logic_v1"
    / "cross_species_test"
    / "test_inference"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 核心输入配置
BEST_EPOCH = 14  # 9  # 你指定的最佳大脑
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

# ========================== 1. 数据加载优化 ==========================
test_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

if not INFERENCE_PATCH_DIR.exists():
    print(f"❌ 错误：找不到切片文件夹 {INFERENCE_PATCH_DIR}，请确保先运行预处理脚本。")
    sys.exit()

inference_ds = datasets.ImageFolder(root=str(INFERENCE_PATCH_DIR), transform=test_tf)

# 使用 DataLoader 加快推理
data_loader = DataLoader(
    inference_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
)

# ========================== 2. 核心功能函数 ==========================


def save_confusion_matrix(labs, preds):
    """保存实战混淆矩阵图"""
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
        f"Final Cross-Species CM (Epoch {BEST_EPOCH})\n[Soft Voting / Leaf Level]"
    )
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label (Voted)")

    img_path = SAVE_DIR / f"final_cm_epoch_{BEST_EPOCH}_soft.png"
    plt.savefig(img_path)
    plt.close()
    print(f"📊 混淆矩阵图已更新: {img_path.name}")


def run_final_inference():
    """
    逻辑说明：
    1. 加载最佳权重，采用 Softmax 获取每个切片的分类概率。
    2. 基于 base_name 累加同张叶片所有切片的信心得分（Soft Voting）。
    3. 比较 [Abiotic_sum, Biotic_sum]，得分高者为最终预测。
    4. 结果以‘追加’模式写入 CSV，保留历史实验记录。
    """
    print(f"\n🚀 启动实战评估 | 权重: Epoch {BEST_EPOCH} | 策略: 信心加权投票")

    # 初始化模型
    model = get_plant_model(num_classes=2).to(DEVICE)
    if not PTH_PATH.exists():
        print(f"❌ 错误：找不到权重文件 {PTH_PATH}")
        return
    model.load_state_dict(torch.load(PTH_PATH, map_location=DEVICE))
    model.eval()

    voting_box = {}  # { "叶片ID": {"conf_sums": [0.0, 0.0], "actual": label} }
    all_samples = inference_ds.samples
    sample_idx = 0

    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Parallel Inferencing"):
            imgs = imgs.to(DEVICE)
            # 得到 logits 后转为概率
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            for i in range(len(probs)):
                img_path, _ = all_samples[sample_idx]
                # 提取文件名作为叶片 ID (逻辑：品种_病害_原名)
                file_stem = Path(img_path).stem
                base_name = "_".join(file_stem.split("_")[:-2])

                if base_name not in voting_box:
                    voting_box[base_name] = {
                        "conf_sums": np.array([0.0, 0.0]),
                        "actual": labels[i].item(),
                    }

                # 累加信心分：解决你说的 90% vs 51% 问题
                voting_box[base_name]["conf_sums"] += probs[i]
                sample_idx += 1

    # 汇总决策
    final_preds, final_labels = [], []
    for data in voting_box.values():
        # voted_res = np.argmax(data["conf_sums"])
        conf = data["conf_sums"] / len(data["hard"])  # 取平均信心
        if conf[1] > 0.9:  # 提高 Biotic 的准入门槛
            voted_res = 1
        else:
            voted_res = 0
        final_preds.append(voted_res)
        final_labels.append(data["actual"])

    # 计算各项关键指标
    acc = 100.0 * (np.array(final_preds) == np.array(final_labels)).mean()
    f1 = f1_score(final_labels, final_preds, average="weighted")
    rec = recall_score(final_labels, final_preds, average="weighted")
    pre = precision_score(final_labels, final_preds, average="weighted")

    # --- 记录数据 (增量写入模式) ---
    new_record = {
        "epoch": BEST_EPOCH,
        "total_leaves": len(final_preds),
        "acc": round(acc, 4),
        "f1": round(f1, 4),
        "recall": round(rec, 4),
        "precision": round(pre, 4),
        "method": "soft_voting",  # 标记这是信心加权法
    }
    df_new = pd.DataFrame([new_record])

    # 核心逻辑：有则增添，无则新建
    file_exists = PERSISTENT_CSV.exists()
    df_new.to_csv(PERSISTENT_CSV, mode="a", index=False, header=not file_exists)

    # 打印最终报告
    print("\n" + "⭐" * 30)
    print(f"🏆 实战总结 (Epoch {BEST_EPOCH})")
    print(f"🎯 准确率: {acc:.2f}% | F1: {f1:.4f}")
    print(f"📝 记录已追加至: {PERSISTENT_CSV.name}")
    print("⭐" * 30)

    # 保存混淆矩阵
    save_confusion_matrix(final_labels, final_preds)


if __name__ == "__main__":
    run_final_inference()
