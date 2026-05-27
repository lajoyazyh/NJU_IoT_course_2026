import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from PyQt5 import QtWidgets

sys.path.append(os.path.dirname(__file__) + '/ui')
from ui.ui import UI


def load_cnn_model():
    """
    载入CNN模型
    :return:
    """
    from model import CNN3
    model = CNN3()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    weight_candidates = [
        os.path.join(project_root, 'models', 'cnn3_best_weights.h5'),
        os.path.join(project_root, 'models', 'cnn2_best_weights.h5')
    ]
    for weight_path in weight_candidates:
        if os.path.exists(weight_path):
            model.load_weights(weight_path)
            return model
    raise FileNotFoundError('No model weights found in models/. Expected cnn3_best_weights.h5 or cnn2_best_weights.h5')
    return model


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QMainWindow()
    model = load_cnn_model()
    ui = UI(form, model)
    form.show()
    sys.exit(app.exec_())
