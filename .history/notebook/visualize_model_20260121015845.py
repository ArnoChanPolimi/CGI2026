import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# --- 1. 绝对精准的路径逻辑 ---
# 因为你在 notebook 文件夹下跑，所以要用 .. 退出到根目录，再进 data
# 路径完全按照你给的：data\train\abiotic\Nutrition_Deficiency_Nutrition Deficiency_003.jpg
img_relative_path = os.path.join(
    "..",
    "data",
    "train",
    "abiotic",
    "Nutrition_Deficiency_Nutrition Deficiency_003.jpg",
)

# 输出到根目录下的 tmp 文件夹
output_folder = os.path.join("..", "tmp")
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "final_comparison.png")

# --- 2. 核心实验：看看你的图片被折腾成什么样了 ---
try:
    img = Image.open(img_relative_path).convert("RGB")
    w, h = img.size
    print(f"✅ 抓到图了！原图尺寸: {w}x{h}")
except FileNotFoundError:
    print(
        f"❌ 还是没找到图！请检查你的项目根目录是不是同时包含 'notebook' 和 'data' 文件夹。"
    )
    print(f"当前尝试路径: {os.path.abspath(img_relative_path)}")
    import sys

    sys.exit(1)

# --- 3. 三种“压缩”逻辑对比 ---
# 逻辑 A：你现在的（Resize 256 + CenterCrop 224）
# 逻辑 B：我说的性能最好的（RandomResizedCrop 224）
# 逻辑 C：保全全貌但模糊的（Pad + Resize 224）

current_logic = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
random_crop_logic = transforms.RandomResizedCrop(224, scale=(0.5, 1.0))


def pad_resize_logic(image):
    max_side = max(image.size)
    p = transforms.Compose(
        [
            transforms.Pad(
                ((max_side - image.size[0]) // 2, (max_side - image.size[1]) // 2),
                fill=0,
            ),
            transforms.Resize((224, 224)),
        ]
    )
    return p(image)


# --- 4. 生成对比图 ---
res1 = current_logic(img)
res2 = random_crop_logic(img)
res3 = pad_resize_logic(img)

plt.figure(figsize=(20, 6))
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title(f"Original\n({w}x{h})")
plt.axis("off")
plt.subplot(1, 4, 2)
plt.imshow(res1)
plt.title("Your Current Logic\n(Center Crop)")
plt.axis("off")
plt.subplot(1, 4, 3)
plt.imshow(res2)
plt.title("Random Crop\n(High Res Local)")
plt.axis("off")
plt.subplot(1, 4, 4)
plt.imshow(res3)
plt.title("Pad & Resize\n(Full Leaf)")
plt.axis("off")

plt.tight_layout()
plt.savefig(output_path)
print(f"🚀 大功告成！对比结果就在: {os.path.abspath(output_path)}")
