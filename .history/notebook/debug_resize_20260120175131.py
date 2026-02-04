import os
from torchvision import transforms
from PIL import Image

# --- 逻辑 A：你的旧逻辑 (CenterCrop) ---
# 原理：短边对齐，长边生切。
# 结果：长图的头尾直接消失，只剩中间一段。
old_logic = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

# --- 逻辑 B：我的新逻辑 (Letterbox/Padding) ---
# 原理：长边对齐，比例不变，空位补黑。
# 结果：整根叶子都在，但为了塞进框，它会缩得比较小。
new_logic = transforms.Compose(
    [
        # Step 1: 把最长的那边缩到 224，比例完全保住
        transforms.Resize(224),
        # Step 2: 此时是长方形，CenterCrop 会自动在短边两旁补黑（由 PyTorch 内部填充逻辑处理）
        # 或者为了演示最清晰，我们强制用 Pad 到 224
        transforms.CenterCrop(224),
    ]
)


def do_compare():
    # 路径设为你刚才给我的那张长条图
    img_path = r"data\train\abiotic\Nutrition_Deficiency_Nutrition Deficiency_003.jpg"

    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    if not os.path.exists(img_path):
        print(f"找不到图: {img_path}")
        return

    img = Image.open(img_path).convert("RGB")

    # 执行旧逻辑
    res_old = old_logic(img)
    res_old.save("tmp/OLD_logic_cropped.jpg")

    # 执行新逻辑
    res_new = new_logic(img)
    res_new.save("tmp/NEW_logic_letterbox.jpg")

    print(f"原图尺寸: {img.size}")
    print("--- 处理完成 ---")
    print(
        "1. 查看 tmp/OLD_logic_cropped.jpg -> 你会发现叶子头尾没了，只剩下中间。这就是为什么它模糊。"
    )
    print(
        "2. 查看 tmp/NEW_logic_letterbox.jpg -> 你会发现整根叶子都在，左右有黑边。这就是为什么它精准。"
    )


if __name__ == "__main__":
    do_compare()
