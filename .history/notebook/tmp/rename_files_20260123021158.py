import os

# 1. 获取当前脚本的绝对路径 (D:\...\notebook\tmp\rename_files.py)
current_script_path = os.path.abspath(__file__)

# 2. 定位到项目根目录 (向上退两级到 notebook 的父目录)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

# 3. 拼接目标文件夹的相对路径
# 路径：raw_data\Wheat\biotic\WheatLeafRust\test\diseased
target_dir = os.path.join(
    project_root, "raw_data", "Wheat", "biotic", "WheatLeafRust", "test", "diseased"
)


def batch_rename():
    # 检查路径是否存在
    if not os.path.exists(target_dir):
        print(f"错误：找不到路径 -> {target_dir}")
        return

    print(f"正在处理目录: {target_dir}")

    count = 0
    # 获取文件夹下所有文件
    for filename in os.listdir(target_dir):
        # 排除已经是 'test_' 开头的文件，防止重复运行
        if filename.startswith("test_"):
            continue

        # 构建完整的新旧路径
        old_path = os.path.join(target_dir, filename)

        # 确保是文件而不是子文件夹
        if os.path.isfile(old_path):
            new_name = f"test_{filename}"
            new_path = os.path.join(target_dir, new_name)

            try:
                os.rename(old_path, new_path)
                count += 1
            except Exception as e:
                print(f"重命名 {filename} 失败: {e}")

    print(f"完成！成功给 {count} 个文件添加了 'test_' 前缀。")


if __name__ == "__main__":
    batch_rename()
