# notebook\test\cross_species_benchmarking_pth12.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from dataset_logic_v1 import get_logic_transforms
from model import get_plant_model

# ================= 路径与配置 =================
MODEL_PATH = Path(r"output\All_crop_train_logic_v1\checkpoint_epoch_12.pth")
OTHER_DATA_DIR = Path(
    r"D:\AA_POLIMI\POLIMI_STUDYING\SEM3\COMMUNICATION_IN_GREEN_INFRASTRUCTURES\CGI_PROJECT\OTHER"
)
RESULT_DIR = Path(r"output\cross_species_test")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
LABEL_MAP = {0: "Abiotic", 1: "Biotic"}

# 1. 加载模型
model = get_plant_model(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
_, test_tf = get_logic_transforms()


def run_evaluation():
    all_summary = []

    # 遍历 01_Lemon, 02_BDHogPlum 等
    for species_folder in sorted(OTHER_DATA_DIR.iterdir()):
        if not species_folder.is_dir():
            continue

        print(f"\n🌿 正在评估物种: {species_folder.name}")
        y_true, y_pred, details = [], [], []

        # 遍历 abiotic (0) 和 biotic (1)
        for target_label, label_name in [(0, "abiotic"), (1, "biotic")]:
            target_path = species_folder / label_name
            if not target_path.exists():
                continue

            # 递归搜索所有子目录下的图片
            img_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG"]:
                img_files.extend(list(target_path.rglob(ext)))

            for img_p in tqdm(img_files, desc=f"  └─ {label_name}", leave=False):
                try:
                    img = Image.open(img_p).convert("RGB")
                    input_tensor = test_tf(img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        output = model(input_tensor)
                        pred = torch.argmax(output, 1).item()
                        prob = torch.softmax(output, 1)[0][pred].item()

                    y_true.append(target_label)
                    y_pred.append(pred)
                    details.append(
                        {
                            "Path": img_p.relative_to(OTHER_DATA_DIR),
                            "True_Label": LABEL_MAP[target_label],
                            "Pred_Label": LABEL_MAP[pred],
                            "Confidence": round(prob, 4),
                            "Is_Correct": target_label == pred,
                        }
                    )
                except Exception as e:
                    continue

        # 2. 计算当前物种的得分
        if y_true:
            acc = accuracy_score(y_true, y_pred)
            all_summary.append(
                {"Species": species_folder.name, "Accuracy": acc, "Total": len(y_true)}
            )

            # 保存该物种的详细错题本
            df_detail = pd.DataFrame(details)
            df_detail.to_csv(
                RESULT_DIR / f"detail_{species_folder.name}.csv", index=False
            )
            print(f"   ✅ 完成! 准确率: {acc:.2%} (样本数: {len(y_true)})")

    # 3. 输出总表
    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(RESULT_DIR / "cross_species_summary.csv", index=False)
    print("\n" + "=" * 30 + "\n最终实战总榜：\n", summary_df.to_string(index=False))


if __name__ == "__main__":
    run_evaluation()
