import cv2
import os

# ====================== 你只需要改这里 ======================
video_path = "data/vedio/2.mp4"  # 你的视频路径
save_dir = "data/image"       # 图片保存文件夹
# ==========================================================

os.makedirs(save_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)

frame_id = 1
counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % 10 == 0:
        # 保存为 1.jpg, 2.jpg...
        cv2.imwrite(f"{save_dir}/{counter}.jpg", frame)
        print(f"已保存 {counter}.jpg")
        counter += 1
    frame_id += 1

cap.release()
print("✅ 视频拆帧完成！")