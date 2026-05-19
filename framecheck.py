import cv2

video_path = r"C:\Users\DeepL_10\Desktop\ドラレコ823\わたろうとする\わたろうとする\渡ろうとする2.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

cap.release()