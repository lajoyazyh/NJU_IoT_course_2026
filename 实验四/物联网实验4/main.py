"""
实验四：实时监控区域入侵检测

打开默认摄像头，动态裁剪画面中心偏上的警戒区，使用课程 CNN 模型判断
是否有人进入，并用 OpenCV Haar 人脸检测作为辅助告警信号。
"""

from datetime import datetime
from pathlib import Path
import time

import cv2
import torch
from PIL import Image
from torchvision import transforms

from model import IoT_CNN


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_model.pth"
LOG_DIR = BASE_DIR / "log"
LOG_PATH = LOG_DIR / "realtime.log"

CAMERA_INDEX = 0
ALARM_INTERVAL_SECONDS = 2
INTRUSION_CLASS = 1
USE_FACE_ASSIST = True

# 动态警戒区比例：宽约 55%，高约 75%，居中并略向上。
ROI_WIDTH_RATIO = 0.55
ROI_HEIGHT_RATIO = 0.75
ROI_CENTER_Y_RATIO = 0.47


def load_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到模型权重文件：{MODEL_PATH}")

    model = IoT_CNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_face_detector():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        print("人脸检测器加载失败，将只使用 CNN 分类结果。")
        return None
    return face_detector


def get_roi_box(frame_w, frame_h):
    roi_w = int(frame_w * ROI_WIDTH_RATIO)
    roi_h = int(frame_h * ROI_HEIGHT_RATIO)
    roi_x = max(0, (frame_w - roi_w) // 2)
    roi_y = max(0, int(frame_h * ROI_CENTER_Y_RATIO - roi_h / 2))

    roi_w = min(roi_w, frame_w - roi_x)
    roi_h = min(roi_h, frame_h - roi_y)
    return roi_x, roi_y, roi_w, roi_h


def preprocess_roi(roi_bgr):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_image = Image.fromarray(roi_rgb)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform(roi_image).unsqueeze(0)


def predict_intrusion(model, roi_bgr, device):
    input_tensor = preprocess_roi(roi_bgr).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)

    pred_class = predicted.item()
    no_person_prob = probabilities[0, 0].item()
    person_prob = probabilities[0, 1].item()
    return pred_class == INTRUSION_CLASS, pred_class, no_person_prob, person_prob


def detect_faces(roi_bgr, face_detector):
    if face_detector is None:
        return []

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )


def write_alarm_log():
    LOG_DIR.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{now}：有人闯入区域\n")


def draw_text(frame, text, origin, color, scale=0.65, thickness=2):
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_status(
    frame,
    roi_box,
    has_intrusion,
    pred_class=None,
    no_person_prob=None,
    person_prob=None,
    face_count=0,
):
    roi_x, roi_y, roi_w, roi_h = roi_box
    if has_intrusion:
        color = (0, 0, 255)
        status_text = "WARNING: INTRUSION DETECTED"
    else:
        color = (0, 180, 0)
        status_text = "SAFE AREA"

    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), color, 2)
    draw_text(frame, status_text, (30, 40), color, scale=0.9)

    if pred_class is not None:
        debug_text = (
            f"CNN Pred: {pred_class} | No: {no_person_prob:.2f} | "
            f"Target: {person_prob:.2f} | Faces: {face_count}"
        )
        draw_text(frame, debug_text, (30, 80), (255, 255, 255))


def draw_face_boxes(frame, roi_box, faces):
    roi_x, roi_y, _, _ = roi_box
    for face_x, face_y, face_w, face_h in faces:
        cv2.rectangle(
            frame,
            (roi_x + face_x, roi_y + face_y),
            (roi_x + face_x + face_w, roi_y + face_y + face_h),
            (255, 0, 0),
            2,
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = load_model(device)
    except FileNotFoundError as exc:
        print(exc)
        return

    face_detector = load_face_detector() if USE_FACE_ASSIST else None

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("摄像头打开失败，请检查摄像头连接或占用情况。")
        return

    print(f"摄像头已开启，当前设备：{device}。按 q 退出。")
    print("调试信息说明：CNN Pred 0=无人，1=有人；Target 是有人类别置信度。")
    last_alarm_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头画面失败。")
                break

            frame_h, frame_w = frame.shape[:2]
            roi_box = get_roi_box(frame_w, frame_h)
            roi_x, roi_y, roi_w, roi_h = roi_box
            roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

            cnn_intrusion, pred_class, no_person_prob, person_prob = predict_intrusion(
                model, roi, device
            )
            faces = detect_faces(roi, face_detector)
            face_count = len(faces)
            has_intrusion = cnn_intrusion or face_count > 0

            draw_face_boxes(frame, roi_box, faces)
            draw_status(
                frame,
                roi_box,
                has_intrusion,
                pred_class,
                no_person_prob,
                person_prob,
                face_count,
            )

            if has_intrusion:
                now = time.time()
                if now - last_alarm_time >= ALARM_INTERVAL_SECONDS:
                    print(
                        "警戒区内检测到目标，触发告警 "
                        f"(CNN Pred={pred_class}, Target={person_prob:.2f}, Faces={face_count})"
                    )
                    write_alarm_log()
                    last_alarm_time = now

            cv2.imshow("IoT Surveillance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("监控已退出。")


if __name__ == "__main__":
    main()
