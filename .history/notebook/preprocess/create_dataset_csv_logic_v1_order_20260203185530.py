import os
from pathlib import Path
import pandas as pd

# ================= 配置区 =================
current_file = Path(__file__).resolve()
ROOT_DIR = current_file.parent.parent.parent

DATA_DIR_NAME = "RAW_DATA"
OUTPUT_FOLDER = ROOT_DIR / "output" / "dataset_index"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# 目标文件名
OUTPUT_CSV = OUTPUT_FOLDER / "dataset_index_letterbox_NoHealthy_v1_NoAloeVera.csv"

# 划分比例
TEST_SIZE = 0.2

ROOT_DIR_STR = str(ROOT_DIR)
OUTPUT_CSV_STR = str(OUTPUT_CSV)
# ==========================================


def create_index_by_sequence():
    full_data_path = os.path.join(ROOT_DIR_STR, DATA_DIR_NAME)
    all_data_list = []

    print(f"开始扫描目录并执行顺序划分 (排除 Healthy): {full_data_path}...")

    if not os.path.exists(full_data_path):
        print(f"错误：找不到路径 {full_data_path}")
        return

    # 1. 遍历作物文件夹
    crop_folders = sorted(os.listdir(full_data_path))  # 排序确保稳定性
    for crop_folder in crop_folders:
        crop_path = os.path.join(full_data_path, crop_folder)
        if not os.path.isdir(crop_path):
            continue

        crop_name = crop_folder.split("_")[-1] if "_" in crop_folder else crop_folder

        # 2. 遍历 abiotic / biotic 类别
        for category in ["abiotic", "biotic"]:
            category_path = os.path.join(crop_path, category)
            if not os.path.exists(category_path):
                continue

            label = 1 if category == "biotic" else 0

            # 3. 遍历具体的病害子类文件夹
            sub_types = sorted(os.listdir(category_path))
            for sub_type in sub_types:
                # 排除 Healthy
                if "healthy" in sub_type.lower():
                    continue

                sub_type_path = os.path.join(category_path, sub_type)
                if not os.path.isdir(sub_type_path):
                    continue

                # 4. 获取该文件夹下所有图片并排序
                files = sorted(
                    [
                        f
                        for f in os.listdir(sub_type_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )

                num_files = len(files)
                if num_files == 0:
                    continue

                # 核心逻辑：计算切分点 (前 80% 训练，后 20% 测试)
                split_idx = int(num_files * (1 - TEST_SIZE))

                # 5. 遍历图片并标记 split
                for i, file in enumerate(files):
                    file_abs_path = os.path.join(sub_type_path, file)
                    rel_path = os.path.relpath(file_abs_path, ROOT_DIR_STR)

                    # 顺序分配
                    split_label = "train" if i < split_idx else "test"

                    all_data_list.append(
                        {
                            "rel_path": rel_path,
                            "crop": crop_name,
                            "category": category,
                            "sub_type": sub_type,
                            "label": label,
                            "split": split_label,
                        }
                    )

    # 生成 DataFrame
    df = pd.DataFrame(all_data_list)

    if df.empty:
        print("错误：未找到任何有效图片！")
        return

    # 保存 CSV
    df.to_csv(OUTPUT_CSV_STR, index=False, encoding="utf-8-sig")

    print("-" * 30)
    print(f"✅ 成功生成顺序划分数据集！文件: {OUTPUT_CSV_STR}")
    print(f"📊 总样本数: {len(df)}")
    print(f"📊 训练集: {len(df[df['split']=='train'])} ({1-TEST_SIZE:.0%})")
    print(f"📊 测试集: {len(df[df['split']=='test'])} ({TEST_SIZE:.0%})")

    # 展示统计结果
    summary = df.groupby(["crop", "sub_type", "split"]).size().unstack().fillna(0)
    print("\n各作物病害详细划分统计 (顺序切分):")
    print(summary.astype(int))


if __name__ == "__main__":
    create_index_by_sequence()
