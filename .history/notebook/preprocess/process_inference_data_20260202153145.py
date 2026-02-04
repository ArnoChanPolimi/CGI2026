import cv2
import os
import numpy as np
from pathlib import Path

# ================= 1. 自动化路径配置 =================
# 获取当前脚本的绝对路径：D:\...\CGI_PROJECT\notebook\preprocess\process_inference_data.py
current_file = Path(__file__).resolve()

# 根据你的项目结构，向上退两层到达项目根目录 (CGI_PROJECT)
# 如果脚本在 notebook\preprocess\，那就是 .parent.parent
PROJECT_ROOT = current_file.parent.parent.parent

# 实战原始大图路径 (相对根目录下的 OTHER)
INPUT_DIR = PROJECT_ROOT / "OTHER"

# 实战切片保存路径 (相对根目录下的 data_processed\inference)
OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "inference"

# 检查路径是否存在，给个提醒
if not INPUT_DIR.exists():
    print(f"❌ 警告：找不到原始数据目录 {INPUT_DIR}")
# =====================================================

# 核心切片参数（保持与训练一致）
TARGET_SIZE = 224
RESIZE_SHORT = 336
STRIDE = 112
SATURATION_THRESHOLD = 30
VALID_RATIO = 0.7
MAX_PATCHES_PER_IMAGE = 8


def process_image_inference(img_path, save_dir, base_name):
    """缩放 -> 切片 -> 过滤 -> 保存"""
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return 0

    h, w = img.shape[:2]
    scale = RESIZE_SHORT / min(h, w)
    img_res = cv2.resize(img, (int(w * scale), int(h * scale)))
    new_h, new_w = img_res.shape[:2]

    count = 0
    for y in range(0, new_h - TARGET_SIZE + 1, STRIDE):
        for x in range(0, new_w - TARGET_SIZE + 1, STRIDE):
            if count >= MAX_PATCHES_PER_IMAGE:
                return count
            patch = img_res[y : y + TARGET_SIZE, x : x + TARGET_SIZE]

            # 饱和度过滤
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            s_channel = hsv[:, :, 1]
            if (
                np.count_nonzero(s_channel > SATURATION_THRESHOLD) / (TARGET_SIZE**2)
            ) > VALID_RATIO:
                count += 1
                # 保持 ID_tile_N 格式，确保分析脚本能投票
                save_name = f"{base_name}_tile_{count}.jpg"
                cv2.imencode(".jpg", patch)[1].tofile(save_dir / save_name)
    return count


def start_inference_processing():
    """遍历 OTHER 文件夹，排除 Healthy，压平分类"""
    total_images = 0

    # 只需要 biotic 和 abiotic
    for category in ["biotic", "abiotic"]:
        target_save_dir = OUTPUT_ROOT / category
        target_save_dir.mkdir(parents=True, exist_ok=True)

        # 遍历品种目录（01_Lemon, 02_BDHogPlum 等）
        for plant_dir in INPUT_DIR.iterdir():
            if not plant_dir.is_dir():
                continue

            # 定位到 biotic 或 abiotic 文件夹
            cat_dir = plant_dir / category
            if not cat_dir.exists():
                continue

            # 遍历具体的病害子目录 (如 Anthracnose, Deficiency 等)
            for disease_dir in cat_dir.iterdir():
                if not disease_dir.is_dir():
                    continue

                # 【核心逻辑】：跳过 Healthy 文件夹
                if "Healthy" in disease_dir.name:
                    print(f"⏩ 跳过健康类目录: {disease_dir}")
                    continue

                # 处理该病害下的所有图片
                for img_p in disease_dir.glob("*"):
                    if img_p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                        continue

                    # 构造 base_name: 品种_具体病名_原文件名
                    # 示例: 01_Lemon_Anthracnose_leaf01
                    base_name = f"{plant_dir.name}_{disease_dir.name}_{img_p.stem}"

                    saved_num = process_image_inference(
                        img_p, target_save_dir, base_name
                    )
                    total_images += 1

        print(f"✅ {category} 类别处理完成。")

    print("-" * 30)
    print(f"🚀 实战数据切片完成！共处理大图: {total_images}")
    print(f"📁 切片存储在: {OUTPUT_ROOT}")


if __name__ == "__main__":
    start_inference_processing()
