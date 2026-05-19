import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image  


# 学習済みCNNモデルをロード

cnn_model = models.mobilenet_v2(weights=None)  # ← pretrainedではなくweights=NoneでOK,重みの部分は後で学習済みの重みを使用する。
cnn_model.classifier[1] = nn.Linear(cnn_model.last_channel, 2)
cnn_model.load_state_dict(torch.load("pedestrian_classifier.pth", map_location="cuda" ))
cnn_model.eval()

device = torch.device("cuda")
cnn_model = cnn_model.to(device)


#  画像前処理の定義

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# YOLOモデルをロード

yolo_model = YOLO(r"C:\Users\DeepL_10\Desktop\pytorch\yolo11x.pt")


#  動画の読み込み

video_path = r"C:\Users\DeepL_10\Desktop\ドラレコ715\FILE250715-161426.MP4"
cap = cv2.VideoCapture(video_path)


#  動画フレームごとの処理

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOで人物検出
    results = yolo_model.predict(frame, imgsz=640, conf=0.6, verbose=False)#resultにyoloで検出した人の情報が入る

    # 各検出された人物ごとにCNNで推論
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls != 0:  # YOLOでperson以外はスキップ
                continue

            # バウンディングボックス座標
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 人物領域を切り抜き
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue

            # OpenCV → PIL 変換
            person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            person_pil = Image.fromarray(person_img_rgb)

            # CNNモデルで推論
            person_tensor = transform(person_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = cnn_model(person_tensor)
                probs = torch.softmax(outputs, dim=1)
                crossing_prob = probs[0][1].item() * 100
                not_crossing_prob = probs[0][0].item() * 100

            # 推論結果を描画
            label = f"{crossing_prob:.1f}% "
            color = (0, 0, 255) if crossing_prob > 50 else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # フレームを表示
    cv2.imshow("YOLO + CNN Pedestrian Prediction", frame)

    # 'q'で終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
