import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train_losses = []
train_accuracies = []


# GPU対応
device = torch.device("cuda")


# データの前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNetV2用サイズ
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# フォルダパス
data_dir = r"C:\Users\DeepL_10\Desktop\dataset"
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# モデルの準備（MobileNetV2）
model = models.mobilenet_v2(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # 転移学習：既存の重みは固定

# 最後の分類層だけ自分のタスクに合わせて変更（2クラス分類）
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)

# 損失関数と最適化
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 学習ループ
for epoch in range(20):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"[{epoch+1}] loss: {running_loss:.3f} acc: {acc:.3f}")


    train_losses.append(running_loss)
    train_accuracies.append(acc)

plt.figure(figsize=(12, 4))

# 損失グラフ
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) 

# 精度グラフ
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.grid()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) 

plt.tight_layout()
plt.show()
print(" 学習完了！")
torch.save(model.state_dict(), "pedestrian_classifier.pth")
print(" モデルを保存しました")


