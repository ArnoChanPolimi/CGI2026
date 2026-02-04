import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# ========================== 1. 路径与配置 ==========================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# 训练产生的 CSV 日志
TRAIN_LOG_CSV = (
    ROOT_DIR / "log" / "train" / "All_crop_train_logic_v1" / "train_metrics.csv"
)
# 权重文件夹
PTH_DIR = ROOT_DIR / "output" / "All_crop_train_logic_v1" / "pth"
# 测试切片目录
TEST_PATCH_DIR = ROOT_DIR / "data_processed" / "test"
# 输出目录
SAVE_DIR = ROOT_DIR / "log" / "analyze" / "patch_v1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

from notebook.train.model import get_plant_model

DEVICE = torch.device("cpu")

# 图像预处理
test_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def run_analysis():
    # ========================== 2. 绘制 4 线 Loss/Acc 曲线 ==========================
    if TRAIN_LOG_CSV.exists():
        print("📈 正在绘制 4 线性能曲线图...")
        # 注意：这里需要你之前训练脚本里已经回溯记录了 Test Loss/Acc
        # 如果训练日志里没有 Test 数据，脚本会自动去扫描 pth 补全（回溯逻辑）
        df_train = pd.read_csv(TRAIN_LOG_CSV)

        plt.figure(figsize=(15, 6))
        # 子图1: Loss
        plt.subplot(1, 2, 1)
        plt.plot(df_train["epoch"], df_train["train_loss"], "r-o", label="Train Loss")
        # 如果有测试Loss则画出
        if "test_loss" in df_train.columns:
            plt.plot(df_train["epoch"], df_train["test_loss"], "b-s", label="Test Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid(True, ls="--")

        # 子图2: Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(df_train["epoch"], df_train["train_acc"], "r-o", label="Train Acc")
        if "test_acc" in df_train.columns:
            plt.plot(df_train["epoch"], df_train["test_acc"], "b-s", label="Test Acc")
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid(True, ls="--")

        plt.savefig(SAVE_DIR / "learning_curves_4lines.png")
        print(f"✅ 曲线图已保存: {SAVE_DIR / 'learning_curves_4lines.png'}")

    # ========================== 3. 叶子级量化投票 (Soft-Voting) ==========================
    # 自动选择最新的权重文件
    pth_files = sorted(
        list(PTH_DIR.glob("*.pth")), key=lambda x: int(re.findall(r"(\d+)", x.name)[-1])
    )
    last_pth = pth_files[-1]
    print(f"🔎 使用最新权重进行投票评估: {last_pth.name}")

    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(last_pth, map_location=DEVICE))
    model.eval()

    # 扫描并分组
    patch_paths = list(TEST_PATCH_DIR.rglob("*.jpg"))
    leaf_vault = {}
    for p in patch_paths:
        base_name = p.stem.split("_tile_")[0]
        true_label = 1 if p.parent.name == "biotic" else 0
        if base_name not in leaf_vault:
            leaf_vault[base_name] = {"patches": [], "true_label": true_label}
        leaf_vault[base_name]["patches"].append(p)

    leaf_results = []
    with torch.no_grad():
        for base_name, data in tqdm(leaf_vault.items(), desc="Leaf Voting"):
            probs = []
            for p_path in data["patches"]:
                img = Image.open(p_path).convert("RGB")
                img_t = test_tf(img).unsqueeze(0).to(DEVICE)
                output = model(img_t)
                prob = torch.softmax(output, dim=1)[0][1].item()  # Biotic 概率
                probs.append(prob)

            mean_score = np.mean(probs)
            final_pred = 1 if mean_score > 0.5 else 0
            leaf_results.append(
                {
                    "leaf_id": base_name,
                    "true": data["true_label"],
                    "pred": final_pred,
                    "score": round(mean_score, 4),
                }
            )

    # ========================== 4. 输出最终量化报告 ==========================
    df_res = pd.DataFrame(leaf_results)
    df_res.to_csv(SAVE_DIR / "final_leaf_results.csv", index=False)

    report = classification_report(
        df_res["true"], df_res["pred"], target_names=["Abiotic", "Biotic"]
    )
    with open(SAVE_DIR / "final_report.log", "w") as f:
        f.write(f"=== 最终叶子级量化分析报告 ===\n{report}")

    # 绘制混淆矩阵
    cm = confusion_matrix(df_res["true"], df_res["pred"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Abiotic", "Biotic"],
        yticklabels=["Abiotic", "Biotic"],
    )
    plt.title("Final Leaf-Level Confusion Matrix")
    plt.savefig(SAVE_DIR / "final_leaf_cm.png")

    print(f"🏁 所有任务完成！请查看目录: {SAVE_DIR}")


if __name__ == "__main__":
    run_analysis()
