import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# --- 1. 相对路径逻辑 (这才是工程标准) ---
# 用 .. 跳出当前的 notebook 文件夹，进入 data 文件夹
img_relative_path = os.path.join(
    "..",
    "data",
    "train",
    "abiotic",
    "Nutrition_Deficiency_Nutrition Deficiency_003.jpg",
)

# 输出到上一层根目录下的 tmp
output_folder = os.path.join("..", "tmp")
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "final_comparison.png")

# --- 2. 核心：加载并处理图片 ---
try:
    img = Image.open(img_relative_path).convert("RGB")
    w, h = img.size
    print(f"✅ 成功加载图！尺寸: {w}x{h}")
except FileNotFoundError:
    print(f"❌ 路径依然报错！请确保你的文件夹结构是：")
    print(f"Project_Root/")
    print(f"  ├── data/ (包含你的图片)")
    print(f"  ├── notebook/ (本脚本 visualize_model.py 在这里)")
    print(f"  └── tmp/ (输出文件夹)")
    import sys

    sys.exit(1)

# --- 3. 三种“尺寸转换”逻辑对比 ---

# 逻辑 1：你现在的代码（Resize + CenterCrop）
# 物理后果：把 3000 长的叶子直接剪掉两头，只留中间 224。
logic_1 = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

# 逻辑 2：性能最好的随机采样（RandomResizedCrop）
# 物理后果：像放大镜一样在 3000 长度里随机抠一块。高清局部。
logic_2 = transforms.RandomResizedCrop(224, scale=(0.5, 1.0))

# 逻辑 3：补齐缩放（Pad + Resize）
# 物理后果：左右补黑边，保全整片叶子，但缩放后叶子只有 20 像素宽。
max_side = max(w, h)
pad_w = (max_side - w) // 2
pad_h = (max_side - h) // 2
logic_3 = transforms.Compose(
    [transforms.Pad((pad_w, pad_h)), transforms.Resize((224, 224))]
)

# --- 4. 绘图对比 ---
plt.figure(figsize=(20, 6))

plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title(f"Original\n({w}x{h})")
plt.axis("off")
plt.subplot(1, 4, 2)
plt.imshow(logic_1(img))
plt.title("Current Logic\n(Info Lost!)")
plt.axis("off")
plt.subplot(1, 4, 3)
plt.imshow(logic_2(img))
plt.title("Proposed: RandomCrop\n(High Res)")
plt.axis("off")
plt.subplot(1, 4, 4)
plt.imshow(logic_3(img))
plt.title("Pad & Resize\n(Too Blurry)")
plt.axis("off")

plt.tight_layout()
plt.savefig(output_path)
print(f"🚀 结果已保存至: {output_path}")
