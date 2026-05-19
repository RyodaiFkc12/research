import torch
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn.functional as F

# 推論に使う画像フォルダ
image_dir = r"C:\Users\DeepL_10\Desktop\test6渡らない"

# クラス名（フォルダ名に合わせて）
class_names = ["wataranai", "wataru"]

device = torch.device("cuda" )

# モデルの準備（MobileNetV2 + 分類層だけ自作）
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("pedestrian_classifier.pth", map_location=device))
model.eval().to(device)

# 前処理（学習時と同じ）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# 対象フォルダ内のすべての画像ファイルで推論
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg')):
        image_path = os.path.join(image_dir, filename)
            # 画像読み込みと前処理
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

            # 推論
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            label = class_names[predicted.item()]
            confidence = probs[0][predicted.item()].item() * 100

        print(f"{filename} → 判定結果：この歩行者は「{label}」と推定されました。")
        print(f" {label} である確率は{confidence:.2f}% です。")

