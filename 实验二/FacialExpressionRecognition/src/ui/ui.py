import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
matplotlib.use("Qt5Agg")
import sys
import os
sys.path.append('../')
from recognition import *

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class PredictWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(str, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, file_name, model):
        super().__init__()
        self.file_name = file_name
        self.model = model

    def run(self):
        try:
            emotion, possibility = predict_expression(self.file_name, self.model)
            self.finished.emit(emotion, possibility)
        except Exception as exc:
            self.failed.emit(str(exc))


class UI(object):
    def __init__(self, form, model):
        self.form = form
        self.worker_thread = None
        self.worker = None
        self.setup_ui(form)
        self.model = model

    def setup_ui(self, form):
        form.setObjectName("Form")
        form.resize(1200, 800)
        # 原图无图时显示的label
        self.label_raw_pic = QtWidgets.QLabel(form)
        self.label_raw_pic.setGeometry(QtCore.QRect(10, 30, 320, 240))
        self.label_raw_pic.setStyleSheet("background-color:#bbbbbb;")
        self.label_raw_pic.setAlignment(QtCore.Qt.AlignCenter)
        self.label_raw_pic.setObjectName("label_raw_pic")
        # 原图下方分割线
        self.line1 = QtWidgets.QFrame(form)
        self.line1.setGeometry(QtCore.QRect(340, 30, 20, 431))
        self.line1.setFrameShape(QtWidgets.QFrame.VLine)
        self.line1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line1.setObjectName("line1")
        # 作者说明label
        self.label_designer = QtWidgets.QLabel(form)
        self.label_designer.setGeometry(QtCore.QRect(20, 700, 180, 40))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_designer.setFont(font)
        self.label_designer.setObjectName("label_designer")
        # 结果布局设置
        self.layout_widget = QtWidgets.QWidget(form)
        self.layout_widget.setGeometry(QtCore.QRect(10, 310, 320, 240))
        self.layout_widget.setObjectName("layoutWidget")
        self.vertical_layout = QtWidgets.QVBoxLayout(self.layout_widget)
        self.vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.vertical_layout.setObjectName("verticalLayout")
        # 右侧水平线
        self.line2 = QtWidgets.QFrame(self.layout_widget)
        self.line2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line2.setObjectName("line2")
        self.vertical_layout.addWidget(self.line2)
        self.horizontal_layout = QtWidgets.QHBoxLayout()
        self.horizontal_layout.setObjectName("horizontalLayout")
        self.pushButton_select_img = QtWidgets.QPushButton(self.layout_widget)
        self.pushButton_select_img.setObjectName("pushButton_2")
        self.horizontal_layout.addWidget(self.pushButton_select_img)
        self.vertical_layout.addLayout(self.horizontal_layout)
        self.graphicsView = QtWidgets.QGraphicsView(form)
        self.graphicsView.setGeometry(QtCore.QRect(360, 210, 800, 500))
        self.graphicsView.setObjectName("graphicsView")
        self.label_result = QtWidgets.QLabel(form)
        self.label_result.setGeometry(QtCore.QRect(361, 21, 71, 16))
        self.label_result.setObjectName("label_result")
        self.label_emotion = QtWidgets.QLabel(form)
        self.label_emotion.setGeometry(QtCore.QRect(715, 21, 71, 16))
        self.label_emotion.setObjectName("label_emotion")
        self.label_emotion.setAlignment(QtCore.Qt.AlignCenter)
        self.label_bar = QtWidgets.QLabel(form)
        self.label_bar.setGeometry(QtCore.QRect(720, 170, 80, 180))
        self.label_bar.setObjectName("label_bar")
        self.line = QtWidgets.QFrame(form)
        self.line.setGeometry(QtCore.QRect(361, 150, 800, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_rst = QtWidgets.QLabel(form)
        self.label_rst.setGeometry(QtCore.QRect(700, 50, 100, 100))
        self.label_rst.setAlignment(QtCore.Qt.AlignCenter)
        self.label_rst.setObjectName("label_rst")

        self.pushButton_select_img.clicked.connect(self.open_file_browser)
        self.retranslate_ui(form)
        QtCore.QMetaObject.connectSlotsByName(form)

    def retranslate_ui(self, form):
        _translate = QtCore.QCoreApplication.translate
        form.setWindowTitle(_translate("Form", "Form"))
        self.label_raw_pic.setText(_translate("Form", "O(∩_∩)O"))
        self.label_designer.setText(_translate("Form", "designed by wss"))
        self.pushButton_select_img.setText(_translate("Form", "选择图像"))
        self.label_result.setText(_translate("Form", "识别结果"))
        self.label_emotion.setText(_translate("Form", "null"))
        self.label_bar.setText(_translate("Form", "概率直方图"))
        self.label_rst.setText(_translate("Form", "Result"))

    def open_file_browser(self):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            return

        file_name, file_type = QtWidgets.QFileDialog.getOpenFileName(caption="选取图片", directory=os.path.join(PROJECT_ROOT, 'input', 'test'),
                                                                     filter="Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)")
        if file_name is not None and file_name != "":
            if not self.show_raw_img(file_name):
                return
            self.pushButton_select_img.setEnabled(False)
            self.pushButton_select_img.setText("识别中...")
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

            self.worker_thread = QtCore.QThread()
            self.worker = PredictWorker(file_name, self.model)
            self.worker.moveToThread(self.worker_thread)

            self.worker_thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.on_predict_success)
            self.worker.failed.connect(self.on_predict_failed)
            self.worker.finished.connect(self.cleanup_worker)
            self.worker.failed.connect(self.cleanup_worker)
            self.worker_thread.start()

    def on_predict_success(self, emotion, possibility):
        self.show_results(emotion, possibility)

    def on_predict_failed(self, error_msg):
        QtWidgets.QMessageBox.critical(self.form, "识别失败", error_msg)

    def cleanup_worker(self):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.pushButton_select_img.setEnabled(True)
        self.pushButton_select_img.setText("选择图像")
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
        self.worker = None

    def show_raw_img(self, filename):
        # 使用 np.fromfile 处理中文路径
        import numpy as np
        try:
            img_array = np.fromfile(filename, dtype=np.uint8)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.form, "图片读取失败", f"无法读取文件:\n{filename}\n\n{exc}")
            return False

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            QtWidgets.QMessageBox.warning(self.form, "图片读取失败", "所选文件不是有效图片，或图片格式不受支持。")
            return False

        frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (320, 240))
        self.label_raw_pic.setPixmap(QtGui.QPixmap.fromImage(
                    QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                                 QtGui.QImage.Format_RGB888)))
        return True

    def show_results(self, emotion, possibility):
        # 显示表情名
        self.label_emotion.setText(QtCore.QCoreApplication.translate("Form", emotion))
        # 显示emoji
        if emotion != 'no':
            icon_path = os.path.join(PROJECT_ROOT, 'assets', 'icons', str(emotion) + '.png')
            img = cv2.imread(icon_path)
            if img is not None:
                frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (100, 100))
                self.label_rst.setPixmap(QtGui.QPixmap.fromImage(
                    QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                                 QtGui.QImage.Format_RGB888)))
            else:
                self.label_rst.setText(QtCore.QCoreApplication.translate("Form", "icon missing"))
        else:
            self.label_rst.setText(QtCore.QCoreApplication.translate("Form", "no result"))
        # 显示直方图
        self.show_bars(list(possibility))

    def show_bars(self, possbility):
        dr = MyFigureCanvas()
        dr.draw_(possbility)
        graphicscene = QtWidgets.QGraphicsScene()
        graphicscene.addWidget(dr)
        self.graphicsView.setScene(graphicscene)
        self.graphicsView.show()

    def get_faces_from_image(self, img_path):
        """
        获取图片中的人脸
        :param img_path:
        :return:
        """
        img, img_gray, faces = face_detect(img_path, 'blazeface')
        if len(faces) == 0:
            return None
        # 遍历每一个脸
        faces_gray = []
        for (x, y, w, h) in faces:
            face_img_gray = img_gray[y:y + h + 10, x:x + w + 10]
            face_img_gray = cv2.resize(face_img_gray, (48, 48))
            faces_gray.append(face_img_gray)
        return faces_gray


class MyFigureCanvas(FigureCanvas):

    def __init__(self, parent=None, width=8, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)

    def draw_(self, possibility):
        x = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        self.axes.bar(x, possibility, align='center')
