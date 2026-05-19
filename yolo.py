import cv2
import time
from ultralytics import YOLO
import torch

# YOLOv11のモデルをロード
model = YOLO(r"C:\Users\DeepL_10\Desktop\pytorch\yolo11x.pt")

# mp4動画ファイルのパス
cap = cv2.VideoCapture(r"C:\Users\DeepL_10\Desktop\渡らない7.mp4")

# FPS計算用の変数
frame_count = 0
frame_times = []

while cap.isOpened():
    torch.cuda.synchronize()
    start_time = time.time()  # フレーム処理開始時間を記録

    ret, frame = cap.read()
    if not ret:
        break

    # YOLOで物体検出を実行
    results = model.predict(frame, save=True, save_crop=True, imgsz=640, conf=0.6, verbose=False)

    # 最初のフレームに描画して表示
    annotated_frame = results[0].plot()

    # ウィンドウに描画した結果を表示
    cv2.imshow("YOLO Detection", annotated_frame)

    # フレーム処理にかかった時間を計算
    torch.cuda.synchronize()
    frame_time = time.time() - start_time
    frame_times.append(frame_time)
    frame_count += 1

    # 'q'で終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 平均FPSを計算
average_fps = frame_count / sum(frame_times)
print(f"Average FPS: {average_fps:.2f}")
print(f"Min frame time: {min(frame_times):.3f} s")
print(f"Max frame time: {max(frame_times):.3f} s")
print(f"Avg frame time: {sum(frame_times)/frame_count:.3f} s")
print(
    "Frame times over 0.050 sec: " +
    ", ".join([f"{t:.3f} s" for i, t in enumerate(frame_times) if t >= 0.05])
)

cap.release()
cv2.destroyAllWindows()
