import torchvision.models as models

# 逻辑：调用官方仓库里的 mobilenet_v2
# weights='DEFAULT' 意味着：不但要下载这个 150 层的结构，
# 还要下载它在百万张图片上练好的“识别能力”（权重参数）。
model = models.mobilenet_v2(weights="DEFAULT")

print("MobileNetV2 下载并加载成功！")
print(model)  # 这行会打印出这 150 层的全部物理结构
