import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ========================== 1. 路径精确配置 ==========================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TEST_PATCH_DIR = ROOT_DIR / "data_processed" / "test"
PTH_PATH = (
    ROOT_DIR
    / "output"
    / "All_crop_train_logic_v1"
    / "pth"
    / "patch_classifier_epoch_10.pth"
)
SAVE_DIR = ROOT_DIR / "log" / "analyze" / "patch_v1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

from notebook.train.model import get_plant_model

DEVICE = torch.device("cpu")

# --- 2. 图像预处理 (必须与训练、切片逻辑严格对齐) ---
test_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def run_scientific_voting():
    # 3. 加载模型
    model = get_plant_model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(PTH_PATH, map_location=DEVICE))
    model.eval()

    # 4. 扫描所有切片，并根据 base_name 分组
    # 文件夹结构是 test/abiotic/ 和 test/biotic/
    patch_paths = list(TEST_PATCH_DIR.rglob("*.jpg"))

    # 核心数据结构：{ "叶子ID": {"patches": [], "true_label": int} }
    leaf_vault = {}

    for p in patch_paths:
        # 提取 base_name: 从 "Tomato_01_tile_1.jpg" 提取出 "Tomato_01"
        base_name = p.stem.split("_tile_")[0]
        true_label = 1 if p.parent.name == "biotic" else 0

        if base_name not in leaf_vault:
            leaf_vault[base_name] = {"patches": [], "true_label": true_label}
        leaf_vault[base_name]["patches"].append(p)

    # 5. 开始推理并计算“平均信心得分”
    leaf_results = []

    print(f"🚀 开始对 {len(leaf_vault)} 片叶子进行科学量化评估...")

    with torch.no_grad():
        for base_name, data in tqdm(leaf_vault.items()):
            biotic_probs = []

            for patch_path in data["patches"]:
                img = Image.open(patch_path).convert("RGB")
                img_t = test_tf(img).unsqueeze(0).to(DEVICE)

                # 获取模型原始输出 (Logits) 并通过 Softmax 转为概率
                output = model(img_t)
                prob = torch.softmax(output, dim=1)
                biotic_prob = prob[0][1].item()  # 获取 Biotic (索引为1) 的信心值
                biotic_probs.append(biotic_prob)

            # 【量化核心】：计算这片叶子所有切片的 Biotic 信心均值
            mean_score = np.mean(biotic_probs)

            # 最终判定：均值过半则为 Biotic
            final_pred = 1 if mean_score > 0.5 else 0

            leaf_results.append(
                {
                    "leaf_id": base_name,
                    "true_label": data["true_label"],
                    "pred_label": final_pred,
                    "confidence_score": round(mean_score, 4),
                    "patch_count": len(data["patches"]),
                }
            )

    # 6. 生成分析报告
    df_res = pd.DataFrame(leaf_results)
    df_res.to_csv(SAVE_DIR / "leaf_level_quantitative_results.csv", index=False)

    # 指志输出 F1/Accuracy
    report = classification_report(
        df_res["true_label"], df_res["pred_label"], target_names=["Abiotic", "Biotic"]
    )

    with open(SAVE_DIR / "scientific_voting_report.log", "w", encoding="utf-8") as f:
        f.write("=== 叶子级量化评估报告 (Soft-Voting) ===\n")
        f.write(f"模型路径: {PTH_PATH.name}\n")
        f.write(report)

    # 混淆矩阵
    cm = confusion_matrix(df_res["true_label"], df_res["pred_label"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Abiotic", "Biotic"],
        yticklabels=["Abiotic", "Biotic"],
    )
    plt.title(f"Leaf-Level Confusion Matrix\n(Based on Mean Confidence Score)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(SAVE_DIR / "leaf_confusion_matrix_soft.png")

    print(f"✅ 评估圆满完成！")
    print(f"📊 报告与量化表已存入: {SAVE_DIR}")


if __name__ == "__main__":
    run_scientific_voting()
