import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


#カスタムデータセット（4フレーム → 1サンプル）

class FourFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None, seq_len=4):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        self.samples = []

        classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for cls_name in classes:
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue

            for seq_folder in os.listdir(cls_path):
                seq_path = os.path.join(cls_path, seq_folder)
                if not os.path.isdir(seq_path):
                    continue

                images = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))
                for i in range(len(images) - seq_len + 1):
                    frame_paths = images[i:i + seq_len]
                    self.samples.append((frame_paths, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        imgs = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        # (3,H,W)×4 → (12,H,W)
        stacked = torch.cat(imgs, dim=0)
        return stacked, label


#  データ前処理・データローダー
data_dir = r"C:\Users\DeepL_10\Desktop\dataset2"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = FourFrameDataset(os.path.join(data_dir, 'train'), transform)
val_dataset   = FourFrameDataset(os.path.join(data_dir, 'val'), transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


#  モデル構築（MobileNetV2を12ch対応に改造）

model = models.mobilenet_v2(pretrained=True)

# 最初のConv層を置き換える
first_conv = model.features[0][0]
new_conv = nn.Conv2d(12, first_conv.out_channels,
                     kernel_size=first_conv.kernel_size,
                     stride=first_conv.stride,
                     padding=first_conv.padding,
                     bias=False)

# 重み初期化：3chの重みを4回コピー
with torch.no_grad():
    for i in range(4):
        new_conv.weight[:, 3*i:3*(i+1), :, :] = first_conv.weight

model.features[0][0] = new_conv

# 分類層を2クラス用に変更
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)


# 学習準備

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

train_losses, train_accuracies = [], []

# 学習ループ

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

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
    train_losses.append(running_loss)
    train_accuracies.append(acc)
    print(f"[{epoch+1}/{num_epochs}] loss: {running_loss:.3f} acc: {acc:.3f}")

print("学習完了！")


# グラフ表示

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.grid()

plt.tight_layout()
plt.show()


# モデル保存

torch.save(model.state_dict(), "pedestrian_4frame_cnn.pth")
print("💾 モデルを保存しました：pedestrian_4frame_cnn.pth")
