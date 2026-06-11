from torchvision.datasets import ImageFolder
from torchvision import transforms

# 1. 加载数据集（和你之前的一样）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ImageFolder(root="data/image", transform=transform)

# 2. 查看 类别 ↔ 标签 对应关系（最关键）
print("===== 文件夹 → 标签 对应 =====")
print(dataset.class_to_idx)  # 输出 {'0':0, '1':1} 就是对的！

# 3. 随机抽查 5 张，看 图片路径 ↔ 标签 是否匹配
print("\n===== 随机抽查 图片-标签 对应 =====")
for i in [0, 10, 20, 30, 40]:  # 抽5个位置检查
    if i < len(dataset):
        img, label = dataset[i]
        img_path = dataset.samples[i][0]  # 拿到图片真实路径
        print(f"路径: {img_path}")
        print(f"对应标签: {label}\n")