import os
import glob
import pandas as pd
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image

#入力（30フレーム画像フォルダ群
ROOT_FOLDER = r"C:\Users\DeepL_10\Desktop\datasetwataranaiLSTM"

#出力
SAVE_ROOT = r"C:\Users\DeepL_10\Desktop\pytorch\LSTMdataset\渡らない"

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#MobileNet
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier = torch.nn.Identity()
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

seq_counter = 1
windows = os.listdir(ROOT_FOLDER)

for win in windows:
    win_path = os.path.join(ROOT_FOLDER, win)
    if not os.path.isdir(win_path):
        continue

    image_paths = sorted(glob.glob(os.path.join(win_path, "*.jpg")))

    #30枚無いフォルダはスキップ
    if len(image_paths) != 30:
        print("skip:", win)
        continue

    print("processing:", win)

    #特徴量抽出
    features_list = []
    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(img)
            feature = feature.cpu().numpy().flatten()

        features_list.append([idx] + feature.tolist())

    #保存先 seqフォルダ
    seq_name = f"seq{seq_counter:03d}"
    save_dir = os.path.join(SAVE_ROOT, seq_name)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(features_list, columns=["frame"] + [f"f{i}" for i in range(1280)])
    csv_path = os.path.join(save_dir, "features.csv")
    df.to_csv(csv_path, index=False)

    print("saved →", csv_path)

    seq_counter += 1

print("保存完了")