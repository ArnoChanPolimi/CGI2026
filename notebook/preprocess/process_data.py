# notebook\preprocess\process_data.py
import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path

# ================= 路径与参数配置 =================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent

# 输入：因为 CSV 里的 rel_path 包含 "RAW_DATA/"，所以输入根目录设为项目根目录
input_root = project_root
output_root = project_root / "data_processed"
# 索引文件路径
CSV_PATH = (
    project_root
    / "output"
    / "dataset_index"
    / "dataset_index_letterbox_NoHealthy_v1.csv"
)

# 核心切片参数
TARGET_SIZE = 224  # 模型需要的输入尺寸
RESIZE_SHORT = 336  # 短边缩放目标
STRIDE = 112  # 步长（224的一半，实现50%重叠）
SATURATION_THRESHOLD = 30  # 饱和度阈值，过滤黑白灰
VALID_RATIO = 0.7  # 必须有70%面积是有色区域
MAX_PATCHES_PER_IMAGE = 8  # 每张叶子最多贡献8个碎片
# =================================================


def process_image(img_path, save_dir, base_name):
    """处理单张图片：缩放 -> 切片 -> 过滤 -> 保存"""
    # 兼容中文路径读取
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return 0

    h, w = img.shape[:2]
    # 短边缩放到 336
    scale = RESIZE_SHORT / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_res = cv2.resize(img, (new_w, new_h))

    count = 0
    # 滑动窗口切割
    for y in range(0, new_h - TARGET_SIZE + 1, STRIDE):
        for x in range(0, new_w - TARGET_SIZE + 1, STRIDE):
            if count >= MAX_PATCHES_PER_IMAGE:
                return count

            patch = img_res[y : y + TARGET_SIZE, x : x + TARGET_SIZE]

            # 饱和度过滤（只看 S 通道）
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            s_channel = hsv_patch[:, :, 1]
            colored_pixels = np.count_nonzero(s_channel > SATURATION_THRESHOLD)

            # 如果叶肉面积合格，则保存
            if (colored_pixels / (TARGET_SIZE * TARGET_SIZE)) > VALID_RATIO:
                count += 1
                # 【关键】保存名：原图ID + 编号，用于后续投票判断
                save_name = f"{base_name}_tile_{count}.jpg"
                save_path = os.path.join(save_dir, save_name)
                cv2.imencode(".jpg", patch)[1].tofile(save_path)
    return count


def start_processing():
    """主控逻辑：根据 CSV 索引进行分流处理"""
    if not CSV_PATH.exists():
        print(f"❌ 错误：找不到 CSV 索引文件 {CSV_PATH}")
        return

    # 读取索引
    df = pd.read_csv(CSV_PATH)
    print(f"🚀 CSV 加载成功，开始处理 {len(df)} 张原图...")

    processed_count = 0
    total_tiles = 0

    for idx, row in df.iterrows():
        # 从 CSV 获取这张图的家谱信息
        rel_path = row["rel_path"]  # "RAW_DATA/..."
        split = row["split"]  # "train" 或 "test"
        category = row["category"]  # "biotic" 或 "abiotic"

        img_full_path = input_root / rel_path

        # 构造二分类文件夹：data_processed/train/biotic 等
        target_dir = output_root / split / category
        target_dir.mkdir(parents=True, exist_ok=True)

        # 获取原图主文件名（base_name）
        base_name = Path(rel_path).stem

        # 执行切片
        saved_num = process_image(img_full_path, target_dir, base_name)

        total_tiles += saved_num
        processed_count += 1

        if processed_count % 100 == 0:
            print(f"已处理: {processed_count}/{len(df)} | 已生成切片: {total_tiles}")

    print("-" * 30)
    print(f"✅ 全部处理完成！")
    print(f"📁 结果路径: {output_root}")
    print(f"📊 总共生成有效切片: {total_tiles}")


if __name__ == "__main__":
    start_processing()
