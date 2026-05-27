import cv2
import argparse
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
import threading

from recognition import generate_faces, face_detect
from model import CNN3
from utils import index2emotion
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=0, help="camera device index usually 0")
    parser.add_argument("--video_path", type=str, default="", help="optional video file path")
    args = parser.parse_args()

    print("[INFO] Loading Model...")
    model = CNN3()
    weights_path = os.path.join(PROJECT_ROOT, "models", "cnn3_best_weights.h5")
    if not os.path.exists(weights_path):
        print(f"[Error] Cannot find weights {weights_path}")
        return
    model.load_weights(weights_path)
    
    from recognition import face_detect
    if args.video_path:
        print(f"[INFO] Opening video file {args.video_path}")
        cap = cv2.VideoCapture(args.video_path)
    else:
        print(f"[INFO] Starting video stream on camera {args.source}...")
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("[Error] Cannot open camera or video file!")
        return

    print("[INFO] Press 'ESC' key to quit the video stream.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用项目原本就带了异常回退保护的 blaze/haar 联合检测逻辑来进行实时画框
        # 但因为原逻辑是收路径这里收帧数据，所以稍微变通一下直接用 cv2 内置的 detect
        # 修复刚写的基于绝对路径的 cascade
        cascade_path = os.path.join(PROJECT_ROOT, 'dataset', 'params', 'haarcascade_frontalface_alt.xml')
        # 避免中文路径引发 opencv C++ cv2.CascadeClassifier 崩溃的方法是先读文件再解析 (但由于 C++ 接口限制，如果是纯中文路径会导致级联器 !empty() 为假)
        # 所以在中文系统上，最安全的办法是用 blazeface
        from blazeface import blaze_detect
        try:
            faces = blaze_detect(frame)
            if faces is None:
                faces = []
        except Exception:
            faces = []
        
        for (x, y, w, h) in faces:
            x, y, w, h = int(x), int(y), int(w), int(h)
            x1 = max(0, x - 10)
            y1 = max(0, y - 10)
            x2 = min(frame.shape[1], x + w + 10)
            y2 = min(frame.shape[0], y + h + 10)
            
            face_img_gray = img_gray[y1:y2, x1:x2]
            if face_img_gray.size == 0:
                continue
                
            try:
                faces_img_gray = generate_faces(face_img_gray)
                results = model.predict(faces_img_gray, verbose=0)
                result_sum = np.sum(results, axis=0).reshape(-1)
                label_index = np.argmax(result_sum, axis=0)
                emotion = index2emotion(label_index, 'en')
                
                # 画框和文字
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                pass

        cv2.imshow("Facial Expression Recognition - Real Time", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
