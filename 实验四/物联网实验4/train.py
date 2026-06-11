import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MyImageFolder
from torchvision import datasets, transforms
from model import IoT_CNN
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import matplotlib.patches as patches
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号



def train(seed=1234):
    # -------------------------- 1.随机种子 --------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # -------------------------- 2.设备 --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- 3.训练集：强增强 ----------------
    train_transform = transforms.Compose([
        # 1. 先缩放到略大，再随机裁剪 → 多尺度+小目标模拟
        transforms.Resize((280, 280)),
        transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),

        # 2. 几何变换：姿态、位置、翻转
        transforms.RandomHorizontalFlip(p=0.5),  # 左右翻转
        transforms.RandomRotation(degrees=(-15, 15)),  # 小角度旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移

        # 3. 颜色/光照：模拟监控逆光、阴影、色温变化
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),

        # 4. 模糊/噪声：模拟低画质、运动模糊
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
        ], p=0.4),

        # 5. 随机灰度：监控黑白/低色彩
        transforms.RandomGrayscale(p=0.1),

        # 6. 最后归一化
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # -------------------------- 4. 加载数据 --------------------------
    # train_dataset = ImageFolder(root="./data/image/train", transform=train_transform)
    # test_dataset = ImageFolder(root="./data/image/test", transform=train_transform)

    # 如果加载数据有问题的话，可以用这个
    train_dataset = MyImageFolder(root="./data/image/train", transform=train_transform)
    test_dataset = MyImageFolder(root="./data/image/test", transform=train_transform)



    # -------------------------- 5. 拆分训练/验证集 --------------------------
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # -------------------------- 6. DataLoader --------------------------
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # -------------------------- 7. 轻量CNN模型 --------------------------
    model = IoT_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # -------------------------- 8. 训练+验证，保留最优权重 --------------------------
    epochs = 20
    best_val_acc = 0.0
    train_loss_list = []
    val_acc_list = []

    print("开始训练...")
    for epoch in range(epochs):
        # 训练
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_loss_list.append(train_loss)

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        val_acc_list.append(val_acc)

        # 保存最优（用验证集）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./model/best_model.pth")
            print(f"✅ 保存最优模型 | 验证集准确率: {val_acc:.2f}%")

        print(f"Epoch [{epoch+1}/{epochs}] 训练损失: {train_loss:.4f} | 验证准确率: {val_acc:.2f}%")

    # -------------------------- 10. 最终测试（只用一次） --------------------------
    print("\n加载最优模型，进行最终测试...")
    model.load_state_dict(torch.load("./model/best_model.pth"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    print(f"🎯 最终测试集准确率: {test_acc:.2f}%")

    # -------------------------- 11. 画损失+准确率曲线 --------------------------
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(val_acc_list, label="Val Acc", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Acc (%)")
    plt.legend()
    plt.show()




# -------------------------- test --------------------------
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------- 加载模型 --------------------------
    model = IoT_CNN().to(device)
    model.load_state_dict(torch.load("./model/best_model.pth"))
    model.eval()

    # -------------------------- 加载数据 --------------------------
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = ImageFolder(root="./data/image/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # -------------------------- 随机找一个图片验证 --------------------------
    image, label = test_dataset[0]


    # -------------------------- 单张图：画警戒区+检测 --------------------------
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5 + 0.5)
    fig, ax = plt.subplots()
    ax.imshow(img)

    # -------------------------- 画警戒区  - -------------------------
    x1, y1 = 80, 80
    w, h = 160, 160
    rect = patches.Rectangle((x1,y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title("IoT监控 - 红色=警戒区")

    roi_img = image[:, y1:y1+h, x1:x1+w]
    roi_img = transforms.Resize((256,256))(roi_img.unsqueeze(0))

    # -------------------------- 检测  - -------------------------
    model.eval()
    with torch.no_grad():
        out = model(roi_img.to(device))
        _, pred = torch.max(out, 1)

    if pred.item() == 1:
        plt.figtext(0.5, 0.01, "警戒区内检测到目标，触发告警", ha="center", color="red")
        print("警戒区内检测到目标，触发告警")
    else:
        plt.figtext(0.5, 0.01, "警戒区内无异常", ha="center", color="green")
        print('"警戒区内无异常"')
    plt.show()


if __name__ == "__main__":
    train()
    test()