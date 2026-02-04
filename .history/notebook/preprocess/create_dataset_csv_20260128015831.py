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

# OUTPUT_CSV = OUTPUT_FOLDER / "dataset_index.csv"
# 建议给它起个新名字，防止和旧的脏数据混淆
OUTPUT_CSV = OUTPUT_FOLDER / "dataset_index_letterbox_v1.csv"

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

    # ================= 修正后的逻辑划分 =================
    # 道理：将分层组设为 "作物_细分类型"，确保每一个具体的文件夹都按比例拆分
    df["stratify_group"] = df["crop"] + "_" + df["sub_type"]

    # 安全检查：找出图片数量太少（小于 2 张）的文件夹，这些文件夹无法进行 8:2 划分
    group_counts = df["stratify_group"].value_counts()
    low_data_groups = group_counts[group_counts < 2]

    if not low_data_groups.empty:
        print("⚠️ 发现以下文件夹图片少于2张，Stratified Split 将报错：")
        print(low_data_groups)
        print("-" * 30)
        # 回退策略：如果某些子文件夹太小，我们退而求其次，按“作物_大类”分层
        print("执行回退逻辑：改按 [作物_大类] 进行分层划分...")
        df["stratify_group"] = df["crop"] + "_" + df["label"].astype(str)

    # 执行划分
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["stratify_group"]
    )
    # ===================================================

    df.loc[train_df.index, "split"] = "train"
    df.loc[test_df.index, "split"] = "test"
    df = df.drop(columns=["stratify_group"])

    # 保存
    df.to_csv(OUTPUT_CSV_STR, index=False, encoding="utf-8-sig")

    print("-" * 30)
    print(f"成功！文件已存至: {OUTPUT_CSV_STR}")
    # 这里的打印也要细化，让你能看清每一个子类是否都分到了数据
    print("各作物详细细分统计：")
    summary = df.groupby(["crop", "sub_type", "split"]).size().unstack().fillna(0)
    print(summary)


if __name__ == "__main__":
    create_index()
