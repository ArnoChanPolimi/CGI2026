import os
import shutil
from pathlib import Path


def prepare_banana_test_set():
    # --- 1. 路径自动定位逻辑 ---
    # 获取当前脚本的绝对路径 (D:\...\notebook\preprocess\preprocess_banana.py)
    script_path = Path(__file__).resolve()

    # 向上跳两级到达项目根目录 (notebook -> 项目根目录)
    project_root = script_path.parents[2]

    # 定义 data 文件夹的相对路径
    data_dir = project_root / "data"
    src_banana = data_dir / "banana"
    dst_challenge = data_dir / "banana_challenge"

    # --- 2. 类别映射关系 ---
    # 根据你的 tree 输出，将子文件夹映射到 abiotic 或 biotic
    mapping = {
        "abiotic": [
            "BANANA DEFICIENCY/POTASSIUM DEFICIENCY",
        ],
        "biotic": [
            "BANANA DISEASE/RESIZED BACTERIAL SOFT ROT",
            "BANANA DISEASE/RESIZED BANANA APHIDS",
            "BANANA DISEASE/RESIZED BANANA FRUIT- SCARRING BEETLE",
            "BANANA DISEASE/RESIZED BLACK SIGATOKA",
            "BANANA DISEASE/RESIZED PANAMA DISEASE",
            "BANANA DISEASE/RESIZED PSEUDOSTEM WEEVIL",
            "BANANA DISEASE/RESIZED YELLOW SIGATOKA",
        ],
    }

    print(f"🚀 项目根目录定位: {project_root}")
    print(f"📂 扫描源路径: {src_banana}")
    print(f"📂 目标路径: {dst_challenge}")

    # --- 3. 清理并创建挑战文件夹 ---
    if dst_challenge.exists():
        print("🧹 清理旧的挑战文件夹...")
        shutil.rmtree(dst_challenge)
    dst_challenge.mkdir(parents=True, exist_ok=True)

    count_total = 0

    # --- 4. 执行文件搬运与重命名 ---
    for target_class, sub_paths in mapping.items():
        class_dir = dst_challenge / target_class
        class_dir.mkdir(exist_ok=True)

        for sub_p in sub_paths:
            # 兼容 Windows/Linux 的路径处理
            full_src_path = src_banana / sub_p

            if not full_src_path.exists():
                print(f"⚠️ 路径不存在，跳过: {sub_p}")
                continue

            print(f"📦 正在归类: {sub_p}")

            # 遍历子文件夹中的图片
            for img_path in full_src_path.iterdir():
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    # 为了防止重名，构造新文件名：文件夹名_原文件名
                    folder_prefix = (
                        sub_p.replace("/", "_").replace("\\", "_").replace(" ", "_")
                    )
                    new_name = f"{folder_prefix}_{img_path.name}"
                    dst_file = class_dir / new_name

                    shutil.copy2(img_path, dst_file)
                    count_total += 1

    print("\n" + "★" * 40)
    print(f"✅ 预处理任务完成！")
    print(f"📊 总计处理香蕉测试图片: {count_total} 张")
    print(f"📍 文件夹已就绪: data/banana_challenge")
    print("★" * 40)


if __name__ == "__main__":
    prepare_banana_test_set()
