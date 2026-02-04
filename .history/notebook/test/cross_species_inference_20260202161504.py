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
from tqdm import tqdm

# ========================== 0. 路径与环境配置 ==========================
# 当前文件：PROJECT_ROOT\notebook\test\cross_species_inference.py
current_file = Path(__file__).resolve()
# .parent 是 test, .parent.parent 是 notebook, .parent.parent.parent 是根目录
PROJECT_ROOT = current_file.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from notebook.train.model import get_plant_model

# 指定输出位置：隔离旧逻辑，专门存放跨品种实战结果
SAVE_DIR = (
    PROJECT_ROOT
    / "output"
    / "All_crop_train_logic_v1"
    / "cross_species_test"
    / "test_inference"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 输入路径
PTH_DIR = PROJECT_ROOT / "output" / "All_crop_train_logic_v1" / "pth"
# 这是你运行处理脚本后生成的实战切片文件夹
INFERENCE_PATCH_DIR = PROJECT_ROOT / "Inference_Data_Processed"
# 原始训练日志（用于对比趋势）
TRAIN_LOG_CSV = (
    PROJECT_ROOT
    / "log"
    / "train"
    / "All_crop_train_logic_v1"
    / "train_metrics_patch_v1.csv"
)

# 持久化 CSV：保存每一轮在实战集上的表现
PERSISTENT_CSV = SAVE_DIR / "cross_species_metrics.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================== 1. 数据加载准备 ==========================
test_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 检查实战切片文件夹是否存在
if not INFERENCE_PATCH_DIR.exists():
    print(f"❌ 错误：找不到实战切片文件夹 {INFERENCE_PATCH_DIR}，请先运行预处理脚本！")
    sys.exit()

# 使用 ImageFolder 加载实战切片
inference_ds = datasets.ImageFolder(root=str(INFERENCE_PATCH_DIR), transform=test_tf)


def evaluate_voted(pth_path, model):
    """【核心逻辑】：对实战数据进行整图投票评估"""
    model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
    model.eval()

    # voting_box: { "叶子ID": {"preds": [], "actual": label} }
    voting_box = {}

    print(f"\n🔬 正在评估权重: {pth_path.name}")
    with torch.no_grad():
        # tqdm 进度条，展示单轮推理进度
        for img_path, label in tqdm(
            inference_ds.samples, desc="Inference Progress", leave=False
        ):
            file_stem = Path(img_path).stem
            # 还原 base_name (格式：品种_病害_原文件名_tile_N)
            # 通过 split 掉最后两个部分 (_tile 和 N) 还原整图 ID
            base_name = "_".join(file_stem.split("_")[:-2])

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

    final_preds, final_labels = [], []
    for data in voting_box.values():
        votes = data["preds"]
        # 绝对数量投票：超过 50% 即判定为 Biotic (1)
        voted_res = 1 if (sum(votes) / len(votes)) > 0.5 else 0
        final_preds.append(voted_res)
        final_labels.append(data["actual"])

    metrics = {
        "test_acc": 100.0 * (np.array(final_preds) == np.array(final_labels)).mean(),
        "test_f1": f1_score(final_labels, final_preds, average="weighted"),
        "test_recall": recall_score(final_labels, final_preds, average="weighted"),
        "test_precision": precision_score(
            final_labels, final_preds, average="weighted"
        ),
    }
    return metrics, final_labels, final_preds


def run_cross_species_analysis():
    # A. 加载训练日志以对比
    df_train = pd.read_csv(TRAIN_LOG_CSV) if TRAIN_LOG_CSV.exists() else pd.DataFrame()
    model = get_plant_model(num_classes=2).to(DEVICE)

    # 获取所有待评估的 pth 文件
    pth_files = sorted(
        list(PTH_DIR.glob("patch_classifier_epoch_*.pth")),
        key=lambda x: int(re.findall(r"epoch_(\d+)", x.name)[0]),
    )

    results = []
    for pth in pth_files:
        epoch = int(re.findall(r"epoch_(\d+)", pth.name)[0])
        m, labs, preds = evaluate_voted(pth, model)

        # 关联训练时的 Loss/Acc 以便绘图对比
        if not df_train.empty:
            train_row = df_train[df_train["epoch"] == epoch]
            if not train_row.empty:
                m.update(
                    {
                        "epoch": epoch,
                        "train_loss": train_row["train_loss"].values[0],
                        "train_acc": train_row["train_acc"].values[0],
                    }
                )
        else:
            m["epoch"] = epoch

        results.append(m)

        # 为第 9 轮（最佳性能预期）单独生成实战混淆矩阵
        if epoch == 9:
            save_confusion_matrix(labs, preds, epoch)

    # B. 保存指标 CSV
    df_res = pd.DataFrame(results).sort_values("epoch")
    df_res.to_csv(PERSISTENT_CSV, index=False)
    print(f"\n📊 实战评估报告已生成：{PERSISTENT_CSV}")

    # C. 绘制实战泛化能力趋势图
    plot_trends(df_res)


def save_confusion_matrix(labs, preds, epoch):
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
    plt.title(f"Cross-Species Confusion Matrix (Epoch {epoch})\n[Voted Leaf Level]")
    plt.ylabel("Actual Label (Ground Truth)")
    plt.xlabel("Predicted Label (Voted)")
    plt.savefig(SAVE_DIR / f"cross_species_cm_epoch_{epoch}.png")
    plt.close()
    print(f"✅ 混淆矩阵图已保存：cross_species_cm_epoch_{epoch}.png")


def plot_trends(df):
    plt.figure(figsize=(10, 6))
    if "train_acc" in df.columns:
        plt.plot(
            df["epoch"], df["train_acc"], "r--", alpha=0.6, label="Training Acc (Patch)"
        )

    plt.plot(
        df["epoch"],
        df["test_acc"],
        "g-o",
        linewidth=2,
        label="Cross-Species Acc (Leaf-Voted)",
    )
    plt.title("Generalization: Training vs Cross-Species Inference")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.savefig(SAVE_DIR / "cross_species_performance_trend.png")
    plt.close()
    print(f"📈 性能对比图已保存：cross_species_performance_trend.png")


if __name__ == "__main__":
    run_cross_species_analysis()
