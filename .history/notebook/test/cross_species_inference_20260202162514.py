import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from torchvision import transforms, datasets
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from tqdm import tqdm

# ========================== 0. 路径与环境配置 ==========================
current_file = Path(__file__).resolve()
# 确保 PROJECT_ROOT 指向 CGI_PROJECT 根目录
PROJECT_ROOT = current_file.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from notebook.train.model import get_plant_model

# 定制输出路径：实战测试结果
SAVE_DIR = (
    PROJECT_ROOT
    / "output"
    / "All_crop_train_logic_v1"
    / "cross_species_test"
    / "test_inference"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 核心输入
BEST_EPOCH = 9
PTH_PATH = (
    PROJECT_ROOT
    / "output"
    / "All_crop_train_logic_v1"
    / "pth"
    / f"patch_classifier_epoch_{BEST_EPOCH}.pth"
)
INFERENCE_PATCH_DIR = PROJECT_ROOT / "Inference_Data_Processed"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================== 1. 预处理与加载 ==========================
test_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

if not INFERENCE_PATCH_DIR.exists():
    print(f"❌ 错误：找不到切片文件夹 {INFERENCE_PATCH_DIR}，请先跑预处理脚本！")
    sys.exit()

inference_ds = datasets.ImageFolder(root=str(INFERENCE_PATCH_DIR), transform=test_tf)

# ========================== 2. 核心逻辑功能 ==========================


def run_final_inference():
    """
    逻辑功能：
    1. 加载指定的 BEST_EPOCH 模型权重。
    2. 遍历所有实战切片，利用文件名中的 base_name 将切片重新归组为“单张叶片”。
    3. 对每张叶片的所有切片预测结果进行“多数投票”。
    4. 对比投票结果与真实标签，计算实战指标并绘制混淆矩阵。
    """
    print(f"🚀 启动最终实战评估 | 目标模型: Epoch {BEST_EPOCH}")

    # 加载模型
    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(PTH_PATH, map_location=DEVICE))
    model.eval()

    voting_box = {}  # 格式: { "叶片ID": {"preds": [], "actual": label} }

    # 推理阶段
    with torch.no_grad():
        for img_path, label in tqdm(
            inference_ds.samples, desc="Processing Patches", unit="patch"
        ):
            # 还原整图 ID (去掉最后两段 _tile 和 _N)
            file_stem = Path(img_path).stem
            base_name = "_".join(file_stem.split("_")[:-2])

            img = (
                test_tf(datasets.folder.default_loader(img_path))
                .unsqueeze(0)
                .to(DEVICE)
            )
            pred = model(img).argmax(1).item()

            if base_name not in voting_box:
                voting_box[base_name] = {"preds": [], "actual": label}
            voting_box[base_name]["preds"].append(pred)

    # 投票决策阶段
    final_preds, final_labels = [], []
    for leaf_id, data in voting_box.items():
        votes = data["preds"]
        # 绝对多数投票逻辑
        voted_res = 1 if (sum(votes) / len(votes)) > 0.5 else 0
        final_preds.append(voted_res)
        final_labels.append(data["actual"])

    # 计算指标
    acc = 100.0 * (np.array(final_preds) == np.array(final_labels)).mean()
    f1 = f1_score(final_labels, final_preds, average="weighted")
    rec = recall_score(final_labels, final_preds, average="weighted")
    pre = precision_score(final_labels, final_preds, average="weighted")

    # 打印最终成绩单
    print("\n" + "=" * 30)
    print(f"🏆 实战评估完成 (Epoch {BEST_EPOCH})")
    print(f"📸 测试叶片总数: {len(final_preds)}")
    print(f"🎯 准确率 (Acc): {acc:.2f}%")
    print(f"🎭 F1 Score: {f1:.4f}")
    print(f"🔍 召回率 (Recall): {rec:.4f}")
    print(f"✨ 精确率 (Precision): {pre:.4f}")
    print("=" * 30)

    # 保存 CSV 结果
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

    # 绘制混淆矩阵
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
    plt.title(
        f"Final Real-World Confusion Matrix (Epoch {BEST_EPOCH})\n[Unit: Single Leaf]"
    )
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label (Voted)")
    plt.savefig(SAVE_DIR / f"final_cm_epoch_{BEST_EPOCH}.png")
    plt.close()
    print(f"📊 混淆矩阵已保存至: {SAVE_DIR}")


if __name__ == "__main__":
    run_final_inference()
