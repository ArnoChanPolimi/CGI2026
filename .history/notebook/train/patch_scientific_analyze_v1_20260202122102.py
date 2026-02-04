import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ========================== 1. 核心路径与配置 ==========================
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from notebook.tools.logger_utils import get_logger
from notebook.train.model import get_plant_model

# 输入路径
SAVE_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1"
PTH_DIR = SAVE_DIR / "pth"
TRAIN_LOG_CSV = (
    ROOT_DIR / "log" / "train" / "All_crop_train_logic_v1" / "train_metrics.csv"
)
INDEX_CSV_PATH = (
    ROOT_DIR / "output" / "dataset_index" / "dataset_index_letterbox_NoHealthy_v1.csv"
)

# 输出路径
ANALYZE_DIR = ROOT_DIR / "log" / "analyze" / "patch_v1"
ANALYZE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
logger = get_logger(str(ANALYZE_DIR), "analysis_report.log")


# ========================== 2. 适配你的索引表 Dataset ==========================
class PatchIndexDataset(Dataset):
    def __init__(self, csv_path, root_dir, split="test", transform=None):
        df = pd.read_csv(csv_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        # 映射：abiotic -> 0, biotic -> 1
        self.label_map = {"abiotic": 0, "biotic": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.root_dir / row["rel_path"]
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row["label"]]
        if self.transform:
            image = self.transform(image)
        return image, label


# ========================== 3. 核心分析逻辑 ==========================
def analyze():
    logger.info(
        "========================== 💡 开始科学阅卷分析 =========================="
    )

    # --- A. 准备数据加载器 ---
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_ds = PatchIndexDataset(
        INDEX_CSV_PATH, ROOT_DIR, split="test", transform=test_tf
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=8)

    # --- B. 曲线回溯 (生成 Test Loss/Acc) ---
    train_history = pd.read_csv(TRAIN_LOG_CSV)
    pth_files = sorted(
        list(PTH_DIR.glob("*.pth")),
        key=lambda x: int(re.findall(r"epoch_(\d+)", x.name)[0]),
    )

    model = get_plant_model(num_classes=2).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = []

    logger.info(f"正在回溯测试集表现 (共 {len(pth_files)} 个权重)...")
    for pth in pth_files:
        epoch = int(re.findall(r"epoch_(\d+)", pth.name)[0])
        model.load_state_dict(torch.load(pth, map_location=DEVICE))
        model.eval()

        t_loss, t_corr, t_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                outs = model(imgs.to(DEVICE))
                loss = criterion(outs, lbls.to(DEVICE))
                t_loss += loss.item()
                t_corr += (torch.argmax(outs, 1) == lbls.to(DEVICE)).sum().item()
                t_total += lbls.size(0)

        test_metrics.append(
            {
                "epoch": epoch,
                "test_loss": t_loss / len(test_loader),
                "test_acc": 100.0 * t_corr / t_total,
            }
        )

    # 合并数据
    full_df = pd.merge(train_history, pd.DataFrame(test_metrics), on="epoch")
    full_df.to_csv(ANALYZE_DIR / "full_performance_metrics.csv", index=False)

    # --- C. 绘制 4 条曲线图 ---
    plt.figure(figsize=(15, 6))

    # 图1: Loss 曲线 (Train vs Test)
    plt.subplot(1, 2, 1)
    plt.plot(full_df["epoch"], full_df["train_loss"], "r-o", label="Train Loss")
    plt.plot(full_df["epoch"], full_df["test_loss"], "b-s", label="Test Loss")
    plt.title("Epoch Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, ls="--")

    # 图2: Acc 曲线 (Train vs Test)
    plt.subplot(1, 2, 2)
    plt.plot(full_df["epoch"], full_df["train_acc"], "r-o", label="Train Acc")
    plt.plot(full_df["epoch"], full_df["test_acc"], "b-s", label="Test Acc")
    plt.title("Epoch Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, ls="--")

    plt.tight_layout()
    plt.savefig(ANALYZE_DIR / "learning_curves_4lines.png")
    logger.info("✅ 4线性能曲线图已保存。")

    # --- D. 针对最新权重 (Epoch 10) 的深度评估 ---
    last_pth = pth_files[-1]
    logger.info(f"📊 正在生成 {last_pth.name} 的详细分类报告...")
    model.load_state_dict(torch.load(last_pth, map_location=DEVICE))

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Final Inference"):
            outs = model(imgs.to(DEVICE))
            all_preds.extend(torch.argmax(outs, 1).cpu().numpy())
            all_labels.extend(lbls.numpy())

    # F1/Precision/Recall 写入日志
    report = classification_report(
        all_labels, all_preds, target_names=["Abiotic", "Biotic"]
    )
    logger.info("\n" + "=" * 20 + " 核心指标报告 " + "=" * 20 + "\n" + report)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Abiotic", "Biotic"],
        yticklabels=["Abiotic", "Biotic"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({last_pth.stem})")
    plt.savefig(ANALYZE_DIR / f"cm_{last_pth.stem}.png")

    logger.info("--- 🚀 分析任务圆满完成！ ---")


if __name__ == "__main__":
    analyze()
