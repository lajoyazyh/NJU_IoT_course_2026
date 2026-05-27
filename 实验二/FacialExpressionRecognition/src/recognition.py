"""
author: Zhou Chen
datetime: 2019/6/19 18:49
desc: 本模块为表情预测处理模块
"""
import os
import cv2
import numpy as np
from utils import index2emotion, expression_analysis, cv2_img_add_text
from blazeface import blaze_detect

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def face_detect(img_path, model_selection="default"):
    """
    检测测试图片的人脸
    :param img_path: 图片的完整路径
    :return:
    """
    # 使用 np.fromfile 处理中文路径
    img_array = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if model_selection == "default":
        cascade_path = os.path.join(PROJECT_ROOT, 'dataset', 'params', 'haarcascade_frontalface_alt.xml')
        # 兼容中文路径：先切换到对应目录，再加载文件名，然后再切回来
        cwd = os.getcwd()
        os.chdir(os.path.dirname(cascade_path))
        face_cascade = cv2.CascadeClassifier(os.path.basename(cascade_path))
        os.chdir(cwd)
        
        faces = face_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30)
        )
    elif model_selection == "blazeface":
        faces = blaze_detect(img)
    else:
        raise NotImplementedError("this face detector is not supported now!!!")

    return img, img_gray, faces


def generate_faces(face_img, img_size=48):
    """
    将探测到的人脸进行增广
    :param face_img: 灰度化的单个人脸图
    :param img_size: 目标图片大小
    :return:
    """
    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img[:, :])
    resized_images.append(face_img[2:45, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))
    # resized_images.append(cv2.flip(face_img[2], 1))
    # resized_images.append(cv2.flip(face_img[3], 1))
    # resized_images.append(cv2.flip(face_img[4], 1))
    resized_images.append(face_img[0:45, 0:45])
    resized_images.append(face_img[2:47, 0:45])
    resized_images.append(face_img[2:47, 2:47])

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images


def predict_expression(img_path, model):
    """
    对图中n个人脸进行表情预测
    :param img_path:
    :return:
    """

    border_color = (0, 0, 0)  # 黑框框
    font_color = (255, 255, 255)  # 白字字

    img, img_gray, faces = face_detect(img_path, 'blazeface')
    if faces is None or len(faces) == 0:
        # BlazeFace 在静态图上有时会漏检，回退到 Haar 提升稳定性
        img, img_gray, faces = face_detect(img_path, 'default')
    if faces is None or len(faces) == 0:
        return 'no', [0, 0, 0, 0, 0, 0, 0, 0]
    # 遍历每一个脸
    emotions = []
    result_possibilitys = []
    img_h, img_w = img_gray.shape[:2]
    for (x, y, w, h) in faces:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w + 10)
        y2 = min(img_h, y + h + 10)
        if x2 <= x1 or y2 <= y1:
            continue

        face_img_gray = img_gray[y1:y2, x1:x2]
        if face_img_gray.size == 0:
            continue
        faces_img_gray = generate_faces(face_img_gray)
        # 预测结果线性加权
        results = model.predict(faces_img_gray, verbose=0)
        result_sum = np.sum(results, axis=0).reshape(-1)
        label_index = np.argmax(result_sum, axis=0)
        emotion = index2emotion(label_index, 'en')

        print(
            "[predict] file={} bbox=({}, {}, {}, {}) mean={:.4f} std={:.4f} probs={}".format(
                os.path.basename(img_path),
                x1,
                y1,
                x2,
                y2,
                float(np.mean(face_img_gray)),
                float(np.std(face_img_gray)),
                np.round(result_sum, 4).tolist()
            )
        )
        
        cv2.rectangle(img, (max(0, x1 - 10), max(0, y1 - 10)), (x2, y2), border_color, thickness=2)
        img = cv2_img_add_text(img, emotion, x1 + 30, y1 + 30, font_color, 20)
        emotions.append(emotion)
        result_possibilitys.append(result_sum)

    if len(emotions) == 0:
        return 'no', [0, 0, 0, 0, 0, 0, 0, 0]
    output_dir = os.path.join(PROJECT_ROOT, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, 'rst.png'), img)
    return emotions[0], result_possibilitys[0]


if __name__ == '__main__':
    from model import CNN3
    model = CNN3()
    model.load_weights('./models/cnn3_best_weights.h5')
    predict_expression('./input/test/happy2.png', model)
