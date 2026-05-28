"""
表情识别模块
加载 CNN3 模型（复用实验二），对单个人脸或多人脸进行表情识别
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, PReLU
from tensorflow.keras.models import Model

# 抑制 TensorFlow 日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 表情标签映射（7 类，与 FER2013 一致）
EMOTION_CN = ['生气', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '中性']
EMOTION_EN = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 模型输入尺寸
IMG_SIZE = 48


def CNN3(input_shape=(48, 48, 1), n_classes=7):
    """
    CNN3 模型结构（与实验二完全一致，但输出类别改为 7 类）
    参考论文: A Compact Deep Learning Model for Robust Facial Expression Recognition
    """
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    # block1
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


class ExpressionRecognizer:
    """表情识别器"""
    
    def __init__(self, model_path=None):
        """
        初始化表情识别器，加载预训练模型
        Args:
            model_path: 模型权重文件路径，默认使用项目中的 cnn3_best_weights.h5
        """
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "models", "cnn3_best_weights.h5")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重文件未找到: {model_path}")
        
        self.model_path = model_path
        self.model = None
        self._warmed_up = False
        self._load_model()
    
    def _load_model(self):
        """加载 CNN3 模型权重"""
        # 尝试加载 7 类模型
        try:
            self.model = CNN3(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=7)
            self.model.load_weights(self.model_path)
            print(f"[ExpressionRecognizer] 模型加载成功 (7类): {self.model_path}")
        except Exception as e:
            # 如果 7 类加载失败，尝试 8 类（实验二原版）
            try:
                print(f"[ExpressionRecognizer] 7类加载失败 ({e})，尝试8类...")
                self.model = CNN3(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=8)
                self.model.load_weights(self.model_path)
                print(f"[ExpressionRecognizer] 模型加载成功 (8类): {self.model_path}")
            except Exception as e2:
                raise RuntimeError(f"模型加载完全失败: {e2}")
    
    def warm_up(self):
        """
        预热模型：做一次空预测以避免首次推理的冷启动延迟
        TensorFlow 首次 predict() 需要编译计算图，可能耗时 5~10 秒
        """
        if self._warmed_up or self.model is None:
            return
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        self.model.predict(dummy, verbose=0)
        self._warmed_up = True
        print("[ExpressionRecognizer] 模型预热完成，首次推理延迟已消除")
    
    def preprocess_face(self, face_bgr, augment=True):
        """
        预处理单个人脸图像
        Args:
            face_bgr: BGR 格式的人脸图像
            augment: 是否使用数据增广（generate_faces 策略）
        Returns:
            face_gray: 灰度化并归一化后的人脸图像 (48, 48, 1)
            或 faces_batch: 增广后的人脸批次 (n, 48, 48, 1)
        """
        if face_bgr is None or face_bgr.size == 0:
            return None
        
        # 转灰度
        if len(face_bgr.shape) == 3 and face_bgr.shape[2] == 3:
            face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_bgr
        
        # 归一化
        face_gray = face_gray.astype(np.float32) / 255.0
        
        if augment:
            # 使用实验二的 generate_faces 增广策略
            return self._generate_faces(face_gray)
        else:
            # 直接缩放到 48x48
            face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_resized = np.expand_dims(face_resized, axis=0)  # 添加 batch 维度
            return face_resized
    
    def _generate_faces(self, face_gray):
        """
        人脸增广（复用实验二的 generate_faces 逻辑）
        Args:
            face_gray: 灰度人脸图像
        Returns:
            resized_images: (n, 48, 48, 1) 的 numpy 数组
        """
        face_gray = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
        
        resized_images = []
        resized_images.append(face_gray[:, :])
        resized_images.append(face_gray[2:45, :] if face_gray.shape[0] > 45 else face_gray)
        resized_images.append(cv2.flip(face_gray[:, :], 1))
        resized_images.append(face_gray[0:45, 0:45] if face_gray.shape[0] > 45 else face_gray)
        resized_images.append(face_gray[2:47, 0:45] if face_gray.shape[0] > 47 else face_gray)
        resized_images.append(face_gray[2:47, 2:47] if face_gray.shape[0] > 47 else face_gray)
        
        for i in range(len(resized_images)):
            resized_images[i] = cv2.resize(resized_images[i], (IMG_SIZE, IMG_SIZE))
            resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
        
        return np.array(resized_images)
    
    def predict_single(self, face_bgr):
        """
        对单个人脸进行表情识别
        Args:
            face_bgr: BGR 格式的人脸图像
        Returns:
            (label_cn, label_en, confidence, probs_dict)
            label_cn: 中文表情标签
            label_en: 英文表情标签
            confidence: 置信度 (0~1)
            probs_dict: 各类别概率字典 {'Angry': 0.1, ...}
        """
        if self.model is None:
            return ("未知", "Unknown", 0.0, {})
        
        faces_batch = self.preprocess_face(face_bgr, augment=True)
        if faces_batch is None:
            return ("未知", "Unknown", 0.0, {})
        
        # 预测（多张增广结果取平均）
        results = self.model.predict(faces_batch, verbose=0)
        result_mean = np.mean(results, axis=0)
        
        # 取最大概率的类别
        label_index = np.argmax(result_mean)
        confidence = float(result_mean[label_index])
        
        # 确保索引不超出范围
        n_classes = self.model.output_shape[-1]
        if label_index >= len(EMOTION_CN):
            label_cn = f"类别{label_index}"
            label_en = f"Class{label_index}"
        else:
            label_cn = EMOTION_CN[label_index]
            label_en = EMOTION_EN[label_index]
        
        # 构建概率字典
        probs_dict = {}
        for i in range(min(n_classes, len(EMOTION_EN))):
            probs_dict[EMOTION_EN[i]] = float(result_mean[i])
        
        return (label_cn, label_en, confidence, probs_dict)
    
    def recognize_all(self, image_bgr, faces):
        """
        对检测到的所有人脸进行表情识别
        Args:
            image_bgr: 原始 BGR 图像
            faces: 人脸框列表 [(x1, y1, x2, y2, confidence), ...]
        Returns:
            results: 识别结果列表
                [{
                    'bbox': (x1, y1, x2, y2, conf),
                    'label_cn': '开心',
                    'label_en': 'Happy',
                    'confidence': 0.95,
                    'probs': {'Angry': 0.01, ...}
                }, ...]
        """
        from face_detector import crop_face
        
        results = []
        
        for face_bbox in faces:
            # 裁剪人脸区域
            face_img = crop_face(image_bgr, face_bbox, margin=5)
            if face_img is None or face_img.size == 0:
                continue
            
            # 表情识别
            label_cn, label_en, confidence, probs = self.predict_single(face_img)
            
            results.append({
                'bbox': face_bbox,
                'label_cn': label_cn,
                'label_en': label_en,
                'confidence': confidence,
                'probs': probs
            })
        
        return results
    
    def recognize_all_fast(self, image_bgr, faces):
        """
        快速批量识别所有人脸的表情（跳过增广，单次批量预测）
        速度比 recognize_all 快约 6 倍，适合实时场景
        Args:
            image_bgr: 原始 BGR 图像
            faces: 人脸框列表 [(x1, y1, x2, y2, confidence), ...]
        Returns:
            results: 识别结果列表（格式同 recognize_all）
        """
        from face_detector import crop_face
        
        if self.model is None or len(faces) == 0:
            return []
        
        # 1. 预处理所有人脸，收集到一个 batch 中
        face_batch_list = []
        valid_faces = []
        
        for face_bbox in faces:
            face_img = crop_face(image_bgr, face_bbox, margin=5)
            if face_img is None or face_img.size == 0:
                continue
            
            # 转灰度 + 归一化 + resize
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_img
            face_gray = face_gray.astype(np.float32) / 255.0
            face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
            face_resized = np.expand_dims(face_resized, axis=-1)  # (48, 48, 1)
            
            face_batch_list.append(face_resized)
            valid_faces.append(face_bbox)
        
        if len(face_batch_list) == 0:
            return []
        
        # 2. 单次批量预测（所有人脸一起推理）
        batch_array = np.array(face_batch_list)  # (N, 48, 48, 1)
        predictions = self.model.predict(batch_array, verbose=0)
        
        # 3. 解析结果
        n_classes = self.model.output_shape[-1]
        results = []
        
        for i, pred in enumerate(predictions):
            label_index = np.argmax(pred)
            confidence = float(pred[label_index])
            
            if label_index >= len(EMOTION_CN):
                label_cn = f"类别{label_index}"
                label_en = f"Class{label_index}"
            else:
                label_cn = EMOTION_CN[label_index]
                label_en = EMOTION_EN[label_index]
            
            probs_dict = {}
            for j in range(min(n_classes, len(EMOTION_EN))):
                probs_dict[EMOTION_EN[j]] = float(pred[j])
            
            results.append({
                'bbox': valid_faces[i],
                'label_cn': label_cn,
                'label_en': label_en,
                'confidence': confidence,
                'probs': probs_dict
            })
        
        return results


# 全局单例
_recognizer_instance = None


def get_recognizer(model_path=None):
    """获取表情识别器单例"""
    global _recognizer_instance
    if _recognizer_instance is None:
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "models", "cnn3_best_weights.h5")
        _recognizer_instance = ExpressionRecognizer(model_path)
    return _recognizer_instance


if __name__ == "__main__":
    print("正在加载表情识别模型...")
    recognizer = get_recognizer()
    print("表情识别模块加载成功！")