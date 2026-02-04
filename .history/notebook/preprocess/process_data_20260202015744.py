# notebook\preprocess\process_data.py
import cv2
import os
import numpy as np
from pathlib import Path

# ================= 相对路径配置 =================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
input_root = project_root / "RAW_DATA"
output_root = project_root / "data_processed"

# 参数设置
TARGET_SIZE = 224
RESIZE_SHORT = 448
STRIDE = 112
SATURATION_THRESHOLD = 30  # 饱和度门槛：低于此值视为黑/白/灰
VALID_RATIO = 0.7  # 必须有70%面积是“有色”的叶肉
MAX_PATCHES_PER_IMAGE = 8  # 每张原图封顶切8张，防止细长叶子把数据撑爆
# ===============================================


def process_image(img_path, save_dir, base_name):
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return

    h, w = img.shape[:2]
    scale = RESIZE_SHORT / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_res = cv2.resize(img, (new_w, new_h))

    count = 0
    # 遍历切片
    for y in range(0, new_h - TARGET_SIZE + 1, STRIDE):
        for x in range(0, new_w - TARGET_SIZE + 1, STRIDE):
            # 限流：一张图切够了就滚去下一张
            if count >= MAX_PATCHES_PER_IMAGE:
                return

            patch = img_res[y : y + TARGET_SIZE, x : x + TARGET_SIZE]

            # --- 认认真真的过滤逻辑：只看饱和度 ---
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            s_channel = hsv_patch[:, :, 1]  # 提取饱和度

            # 数数有多少像素是有颜色的 (S > 30)
            colored_pixels = np.count_nonzero(s_channel > SATURATION_THRESHOLD)

            # 只有有色面积 > 70%，才认为是合格的叶肉切片
            if (colored_pixels / (TARGET_SIZE * TARGET_SIZE)) > VALID_RATIO:
                count += 1
                save_name = f"{base_name}_tile_{count}.jpg"
                save_path = os.path.join(save_dir, save_name)
                cv2.imencode(".jpg", patch)[1].tofile(save_path)


def start_processing():
    if not input_root.exists():
        print(f"❌ 错误：找不到原始数据路径 {input_root}")
        return

    # 自动排除掉 AloeVera 文件夹（如果你还没手动删的话）
    ignored_folders = ["09_AloeVera"]

    print(f"🚀 开始处理！\n输入路径: {input_root}\n输出路径: {output_root}")

    for crop_folder in os.listdir(input_root):
        if crop_folder in ignored_folders:
            continue

        crop_path = input_root / crop_folder
        if not crop_path.is_dir():
            continue

        for category in ["abiotic", "biotic"]:
            category_path = crop_path / category
            if not category_path.exists():
                continue

            target_dir = output_root / crop_folder / category
            target_dir.mkdir(parents=True, exist_ok=True)

            for sub_dir in os.listdir(category_path):
                sub_dir_path = category_path / sub_dir
                if not sub_dir_path.is_dir():
                    continue

                print(f"正在清洗: {crop_folder} -> {category} -> {sub_dir}")

                for file in os.listdir(sub_dir_path):
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        process_image(
                            sub_dir_path / file,
                            str(target_dir),
                            (sub_dir_path / file).stem,
                        )

    print("✅ 全部处理完成！这次出来的图绝对没有黑边白边了。")


if __name__ == "__main__":
    start_processing()
