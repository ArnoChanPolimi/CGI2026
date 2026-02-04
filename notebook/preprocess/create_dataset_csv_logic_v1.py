# notebook\preprocess\create_dataset_csv_logic_v1.py
# This script creates a dataset CSV file excluding 'Healthy' folders.
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ================= 配置区 =================
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent

DATA_DIR_NAME = "RAW_DATA"
OUTPUT_FOLDER = ROOT_DIR / "output" / "dataset_index"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# 明确新文件名，逻辑对齐
# OUTPUT_CSV = OUTPUT_FOLDER / "dataset_index_letterbox_NoHealthy_v1.csv"
OUTPUT_CSV = OUTPUT_FOLDER / "dataset_index_letterbox_NoHealthy_v1_NoAloeVera.csv"

SEED = 42
TEST_SIZE = 0.2

ROOT_DIR_STR = str(ROOT_DIR)
OUTPUT_CSV_STR = str(OUTPUT_CSV)
# ==========================================


def create_index():
    full_data_path = os.path.join(ROOT_DIR_STR, DATA_DIR_NAME)
    data_list = []

    print(f"开始扫描目录 (排除 Healthy 文件夹): {full_data_path}...")

    if not os.path.exists(full_data_path):
        print(f"错误：找不到路径 {full_data_path}")
        return

    for crop_folder in os.listdir(full_data_path):
        crop_path = os.path.join(full_data_path, crop_folder)
        if not os.path.isdir(crop_path):
            continue

        crop_name = crop_folder.split("_")[-1] if "_" in crop_folder else crop_folder

        for category in ["abiotic", "biotic"]:
            category_path = os.path.join(crop_path, category)
            if not os.path.exists(category_path):
                continue

            label = 1 if category == "biotic" else 0

            for sub_type in os.listdir(category_path):
                # ================= 核心修改：精准剔除 Healthy =================
                # 不管是大写 Healthy 还是小写 healthy，只要文件夹叫这个名字，直接跳过
                if "healthy" in sub_type.lower():
                    continue
                # ============================================================

                sub_type_path = os.path.join(category_path, sub_type)
                if not os.path.isdir(sub_type_path):
                    continue

                for file in os.listdir(sub_type_path):
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        file_abs_path = os.path.join(sub_type_path, file)
                        rel_path = os.path.relpath(file_abs_path, ROOT_DIR_STR)

                        data_list.append(
                            {
                                "rel_path": rel_path,
                                "crop": crop_name,
                                "category": category,
                                "sub_type": sub_type,
                                "label": label,
                            }
                        )

    df = pd.DataFrame(data_list)
    if df.empty:
        print("错误：剔除 Healthy 后未找到任何图片！请检查目录结构。")
        return

    # 分层逻辑
    df["stratify_group"] = df["crop"] + "_" + df["sub_type"]
    group_counts = df["stratify_group"].value_counts()
    low_data_groups = group_counts[group_counts < 2]

    if not low_data_groups.empty:
        print("⚠️ 某些病害子类样本过少，执行分层回退逻辑...")
        df["stratify_group"] = df["crop"] + "_" + df["label"].astype(str)

    # 划分
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["stratify_group"]
    )

    df.loc[train_df.index, "split"] = "train"
    df.loc[test_df.index, "split"] = "test"
    df = df.drop(columns=["stratify_group"])

    # 保存新 CSV
    df.to_csv(OUTPUT_CSV_STR, index=False, encoding="utf-8-sig")

    print("-" * 30)
    print(f"✅ 成功生成纯净数据集！文件: {OUTPUT_CSV_STR}")
    print(f"📊 总样本数 (不含 Healthy): {len(df)}")
    print(f"📊 类别统计: \n{df['category'].value_counts()}")

    summary = df.groupby(["crop", "sub_type", "split"]).size().unstack().fillna(0)
    print("\n各作物病害详细划分统计：")
    print(summary)


if __name__ == "__main__":
    create_index()
