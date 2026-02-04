# notebook\preprocess.py
import cv2
import os
import numpy as np


def process_abiotic():
    # 路径根据你的 VSCode 结构调整
    src_path = r"Abiotic_Stress\rice_plant_lacks_nutrients\Nitrogen(N)"
    dst_path = r"data\processed\Abiotic"
    os.makedirs(dst_path, exist_ok=True)

    valid_exts = (".jpg", ".png", ".jpeg")
    files = [f for f in os.listdir(src_path) if f.lower().endswith(valid_exts)]

    print(f"开始处理 Abiotic 路径，共 {len(files)} 张图片...")

    for filename in files:
        img = cv2.imread(os.path.join(src_path, filename))
        if img is None:
            continue

        h, w = img.shape[:2]
        side = min(h, w)  # 窗口大小：298
        # 步长：重叠20%，移动80% (238 像素)
        stride = int(side * 0.8)

        base_name = os.path.splitext(filename)[0]
        count = 0

        # 逻辑：滑动切片
        if w > h:  # 横向长图
            # 从 0 开始，每隔 stride 像素切一刀
            for x in range(0, w - side + 1, stride):
                tile = img[:, x : x + side]
                save_img(tile, dst_path, f"{base_name}_tile_{count}")
                count += 1
            # 逻辑：解决除不尽，末端回溯
            last_tile = img[:, w - side : w]
            save_img(last_tile, dst_path, f"{base_name}_end")

        else:  # 纵向长图
            for y in range(0, h - side + 1, stride):
                tile = img[y : y + side, :]
                save_img(tile, dst_path, f"{base_name}_tile_{count}")
                count += 1
            last_tile = img[h - side : h, :]
            save_img(last_tile, dst_path, f"{base_name}_end")

    print(f"Abiotic 处理完成，生成了大量 224x224 无失真样本。")


def save_img(img, folder, name):
    # 统一缩放到 224x224。使用三次插值保证质量。
    final = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(folder, f"{name}.jpg"), final)


if __name__ == "__main__":
    process_abiotic()
