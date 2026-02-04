# notebook\preprocess\create_dataset_csv.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ================= 配置区 =================
# 指向你的项目根目录 (相对于本脚本所在位置或使用绝对路径)
ROOT_DIR = r"D:\AA_POLIMI\POLIMI_STUDYING\SEM3\COMMUNICATION IN GREEN INFRASTRUCTURES\CGI_PROJECT"
DATA_DIR_NAME = "RAW_DATA"  # 你的原始数据文件夹名
OUTPUT_CSV = os.path.join(ROOT_DIR, "dataset_index.csv")
SEED = 42  # 随机种子，确保每次运行结果一致
TEST_SIZE = 0.2  # 20% 作为测试集
# ==========================================


def create_index():
    full_data_path = os.path.join(ROOT_DIR, DATA_DIR_NAME)
    data_list = []

    print(f"开始扫描目录: {full_data_path}...")

    # 遍历 作物层 (如 01_Chilli)
    for crop_folder in os.listdir(full_data_path):
        crop_path = os.path.join(full_data_path, crop_folder)
        if not os.path.isdir(crop_path):
            continue

        # 提取干净的作物名 (去掉数字前缀)
        crop_name = crop_folder.split("_")[-1] if "_" in crop_folder else crop_folder

        # 遍历 大类层 (abiotic/biotic)
        for category in ["abiotic", "biotic"]:
            category_path = os.path.join(crop_path, category)
            if not os.path.exists(category_path):
                continue

            label = 1 if category == "biotic" else 0

            # 遍历 细分层 (如 Healthy, Rust)
            for sub_type in os.listdir(category_path):
                sub_type_path = os.path.join(category_path, sub_type)
                if not os.path.isdir(sub_type_path):
                    continue

                # 遍历所有照片
                for file in os.listdir(sub_type_path):
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        # 重点：计算【相对路径】
                        # 结果类似: RAW_DATA/01_Chilli/abiotic/Healthy/img.jpg
                        rel_path = os.path.relpath(
                            os.path.join(sub_type_path, file), ROOT_DIR
                        )

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
        print("错误：未找到任何图片，请检查路径配置！")
        return

    # --- 执行逻辑划分 (Stratified Split) ---
    # 我们根据 'crop' 和 'label' 的组合进行分层，确保每个作物里的 0和1 比例都被平衡划分
    df["stratify_group"] = df["crop"] + "_" + df["label"].astype(str)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["stratify_group"]
    )

    # 在原表中标注 split 属性
    df.loc[train_df.index, "split"] = "train"
    df.loc[test_df.index, "split"] = "test"

    # 删除辅助列并保存
    df = df.drop(columns=["stratify_group"])
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("-" * 30)
    print(f"成功生成清单！总计照片: {len(df)} 张")
    print(f"训练集: {len(train_df)} 张 | 测试集: {len(test_df)} 张")
    print(f"清单文件已保存至: {OUTPUT_CSV}")
    print("-" * 30)
    print("各作物分布统计:")
    print(df.groupby(["crop", "split"]).size().unstack().fillna(0))


if __name__ == "__main__":
    create_index()
