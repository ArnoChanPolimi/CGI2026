import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm  # 导入进度条库

# ================= 1. 自动化路径配置 =================
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent

INPUT_DIR = PROJECT_ROOT / "other"  # 请确认你的文件夹名是大写 OTHER 还是小写 other
OUTPUT_ROOT = PROJECT_ROOT / "Inference_Data_Processed"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

if not INPUT_DIR.exists():
    print(f"❌ 错误：找不到原始数据目录 {INPUT_DIR}")
    exit()

# 核心切片参数
TARGET_SIZE = 224
RESIZE_SHORT = 336
STRIDE = 112
SATURATION_THRESHOLD = 30
VALID_RATIO = 0.7
MAX_PATCHES_PER_IMAGE = 8


def process_image_inference(img_path, save_dir, base_name):
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

            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            s_channel = hsv[:, :, 1]
            if (
                np.count_nonzero(s_channel > SATURATION_THRESHOLD) / (TARGET_SIZE**2)
            ) > VALID_RATIO:
                count += 1
                save_name = f"{base_name}_tile_{count}.jpg"
                cv2.imencode(".jpg", patch)[1].tofile(save_dir / save_name)
    return count


def start_inference_processing():
    print(f"🚀 开始实战数据预处理...")

    # 1. 先扫描所有任务，统计总数以便显示总进度
    all_tasks = []
    for category in ["biotic", "abiotic"]:
        for plant_dir in INPUT_DIR.iterdir():
            if not plant_dir.is_dir():
                continue
            cat_dir = plant_dir / category
            if not cat_dir.exists():
                continue
            for disease_dir in cat_dir.iterdir():
                if not disease_dir.is_dir() or "Healthy" in disease_dir.name:
                    continue
                # 收集该目录下所有图片路径
                images = [
                    p
                    for p in disease_dir.glob("*")
                    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ]
                for img_p in images:
                    all_tasks.append(
                        (img_p, category, plant_dir.name, disease_dir.name)
                    )

    print(f"📊 扫描完成！共计 {len(all_tasks)} 张大图需要处理（已排除 Healthy）。")

    # 2. 使用 tqdm 显示进度条
    total_tiles = 0
    # 用 tqdm 包裹列表，它会自动计算进度
    for img_p, category, plant_name, disease_name in tqdm(
        all_tasks, desc="Processing Images", unit="img"
    ):
        target_save_dir = OUTPUT_ROOT / category
        target_save_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{plant_name}_{disease_name}_{img_p.stem}"
        saved_num = process_image_inference(img_p, target_save_dir, base_name)
        total_tiles += saved_num

    print("-" * 30)
    print(f"✅ 处理完成！")
    print(f"📁 存储路径: {OUTPUT_ROOT}")
    print(f"📊 总共生成有效切片: {total_tiles}")


if __name__ == "__main__":
    start_inference_processing()
