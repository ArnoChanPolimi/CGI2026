import os
from PIL import Image
from torchvision import transforms

# 1. 严格执行逻辑：缩256 -> 切224 -> 四向旋转
# 这种写法确保了在 224 正方形内做无像素损失的直角旋转
test_transform = transforms.Compose(
    [
        transforms.Resize(256),  # 保护长宽比，短边缩放至256
        transforms.CenterCrop(224),  # 利用你说的“白边”缓冲，裁掉背景锁定主体
        transforms.RandomChoice(
            [
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
            ]
        ),
    ]
)


def verify_logic(img_path, save_dir="tmp"):
    # 路径健壮性处理
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 使用规范化的路径读取
    try:
        # 这里直接使用传入的路径，确保路径格式正确
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        print(f"✅ 成功读取！原始尺寸: {w}x{h}")
    except FileNotFoundError:
        print(f"❌ 找不到文件！请检查路径：{os.path.abspath(img_path)}")
        return

    # 生成 5 张图，观察旋转和剪裁的物理效果
    for i in range(5):
        processed_img = test_transform(img)
        save_path = os.path.join(save_dir, f"processed_sample_{i}.jpg")
        processed_img.save(save_path)
        print(f"🚀 已生成处理图 {i}: {save_path} | 尺寸: {processed_img.size}")


if __name__ == "__main__":
    # 路径完全按照你的要求写死，不再胡扯
    target_image = os.path.join(
        "data", "train", "abiotic", "Nutrition_Deficiency_Nutrition Deficiency_003.jpg"
    )

    verify_logic(target_image)
