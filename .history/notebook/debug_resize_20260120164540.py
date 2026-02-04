import os
from torchvision import transforms
from PIL import Image

# 1. 定死逻辑：等比例缩放 + 黑色补齐
process_logic = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ]
)


def run():
    # 手动填入你刚才给我的路径
    img_path = r"data\original_dataset_Chilli_Leaf\Cercospora_Leaf_Spot\Cercospora Leaf Spot_002.jpg"

    # 创建 tmp 文件夹
    save_dir = "tmp"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(img_path):
        print(f"找不到文件: {img_path}")
        return

    # 处理并保存
    img = Image.open(img_path).convert("RGB")
    processed_img = process_logic(img)

    save_path = os.path.join(save_dir, "check_long_leaf.jpg")
    processed_img.save(save_path)

    print(f"处理完成！原图尺寸: {img.size}")
    print(f"结果已存至: {save_path}")


if __name__ == "__main__":
    run()
