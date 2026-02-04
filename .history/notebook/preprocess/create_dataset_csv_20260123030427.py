# notebook\preprocess\create_dataset_csv.py
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ================= 配置区 =================
current_file = Path(__file__).resolve()

# 确保跳级逻辑正确：preprocess -> notebook -> CGI_PROJECT
# 这样 ROOT_DIR 始终是你的项目根目录
ROOT_DIR = current_file.parent.parent.parent

DATA_DIR_NAME = "RAW_DATA"
OUTPUT_FOLDER = ROOT_DIR / "output" / "dataset_index"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_FOLDER / "dataset_index.csv"

SEED = 42
TEST_SIZE = 0.2

# 统一将 Path 对象转为字符串，确保函数内部所有 os.path 逻辑调用一致
# 这样你就不会发现“定义了却没用到”的情况
ROOT_DIR_STR = str(ROOT_DIR)
OUTPUT_CSV_STR = str(OUTPUT_CSV)
# ==========================================


def create_index():
    # 使用转换后的字符串，逻辑闭环
    full_data_path = os.path.join(ROOT_DIR_STR, DATA_DIR_NAME)
    data_list = []

    print(f"开始扫描目录: {full_data_path}...")

    if not os.path.exists(full_data_path):
        print(f"错误：找不到路径 {full_data_path}，请检查文件夹名称！")
        return

    for crop_folder in os.listdir(full_data_path):
        crop_path = os.path.join(full_data_path, crop_folder)
        if not os.path.isdir(crop_path):
            continue

        # 提取作物名 (如 01_Chilli -> Chilli)
        crop_name = crop_folder.split("_")[-1] if "_" in crop_folder else crop_folder

        for category in ["abiotic", "biotic"]:
            category_path = os.path.join(crop_path, category)
            if not os.path.exists(category_path):
                continue

            label = 1 if category == "biotic" else 0

            for sub_type in os.listdir(category_path):
                sub_type_path = os.path.join(category_path, sub_type)
                if not os.path.isdir(sub_type_path):
                    continue

                for file in os.listdir(sub_type_path):
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        # 这里统一使用 ROOT_DIR_STR，逻辑达成一致
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
        print("错误：未找到任何图片！")
        return

    # 逻辑划分
    df["stratify_group"] = df["crop"] + "_" + df["label"].astype(str)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["stratify_group"]
    )

    df.loc[train_df.index, "split"] = "train"
    df.loc[test_df.index, "split"] = "test"
    df = df.drop(columns=["stratify_group"])

    # 使用 OUTPUT_CSV_STR，逻辑再次闭环
    df.to_csv(OUTPUT_CSV_STR, index=False, encoding="utf-8-sig")

    print("-" * 30)
    print(f"成功！文件已存至: {OUTPUT_CSV_STR}")
    print(
        f"各作物分布情况：\n{df.groupby(['crop', 'split']).size().unstack().fillna(0)}"
    )


if __name__ == "__main__":
    create_index()
