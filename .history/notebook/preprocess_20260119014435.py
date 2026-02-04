# notebook\preprocess.py
import cv2
import os


def process_nitrogen_special():
    # 你的指定路径
    src_path = r"Abiotic_Stress\rice_plant_lacks_nutrients\Nitrogen(N)"
    dst_path = r"data\raw_Aug\rice_Abiotic"

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    # 遍历该路径下所有图
    for filename in os.listdir(src_path):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img = cv2.imread(os.path.join(src_path, filename))
        if img is None:
            continue

        h, w = img.shape[:2]

        # 底层逻辑：针对长条图进行“无损局部采样”
        # 我们以短边为基准，但在长边上每隔一段距离就切一个方块
        side = min(h, w)
        # 增加采样密度：每移动 0.5 倍的边长就切一张，这样能抓到更多变色细节
        stride = int(side * 0.5)

        count = 0
        if w > h:  # 如果是横向长叶子
            for x in range(0, w - side + 1, stride):
                crop = img[:, x : x + side]
                save_and_resize(crop, dst_path, f"N_long_{filename}_{count}")
                count += 1
        else:  # 如果是纵向长叶子
            for y in range(0, h - side + 1, stride):
                crop = img[y : y + side, :]
                save_and_resize(crop, dst_path, f"N_long_{filename}_{count}")
                count += 1

        print(f"处理完成：{filename}，生成了 {count} 张训练切片")


def save_and_resize(img, folder, name):
    # 统一缩放到 224x224，对接 MobileNetV2
    final = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(folder, f"{name}.jpg"), final)


if __name__ == "__main__":
    process_nitrogen_special()
