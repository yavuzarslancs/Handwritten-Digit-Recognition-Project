import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import os
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model



class MainWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        
        self.loaded_model = load_model('rakam_tanima_(digit_recognition).h5')

        self.initUI()
    
    def initUI(self):
        
        self.container = QtWidgets.QVBoxLayout()
        self.container.setContentsMargins(0, 0, 0, 0)

        
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(450, 450)
        canvas.fill(QtGui.QColor("black"))
        self.label.setPixmap(canvas)
        self.last_x, self.last_y = None, None

       
        self.prediction = QtWidgets.QLabel('Prediction: ...')
        self.prediction.setFont(QtGui.QFont('Monospace', 20))

        
        self.button_clear = QtWidgets.QPushButton('CLEAR')
        self.button_clear.clicked.connect(self.clear_canvas)

        
        self.container.addWidget(self.label)
        self.container.addWidget(self.prediction, alignment=QtCore.Qt.AlignHCenter)
        self.container.addWidget(self.button_clear, alignment=QtCore.Qt.AlignHCenter)

        self.setLayout(self.container)
    
    def clear_canvas(self):
       
        self.label.pixmap().fill(QtGui.QColor('#000000'))
        self.update()
        self.prediction.setText('Prediction: ...')

   
    def mouseMoveEvent(self, e):
        if self.last_x is None:  
            self.last_x = e.x()
            self.last_y = e.y()
            return  

        painter = QtGui.QPainter(self.label.pixmap())

        p = painter.pen()
        p.setWidth(20)
        self.pen_color = QtGui.QColor('#FFFFFF')
        p.setColor(self.pen_color)
        painter.setPen(p)

        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        
        self.last_x = None
        self.last_y = None

        s = self.label.pixmap().toImage().bits().asarray(450 * 450 * 4)
        arr = np.frombuffer(s, dtype=np.uint8).reshape((450, 450, 4))
        arr = np.array(ImageOps.grayscale(ImageOps.fit(Image.fromarray(arr), (28, 28), method=0, bleed=0.0)))
        arr = (arr/255.0).reshape(1, 28, 28, 1)  
        self.prediction.setText('Prediction: '+str(np.argmax(self.loaded_model.predict(arr))))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        
        self.mainWidget = MainWidget()
        self.setCentralWidget(self.mainWidget)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    mainApp = MainWindow()
    mainApp.setWindowTitle('Tahmin Edici(Digit Predicter)')
    mainApp.show()
    sys.exit(app.exec_())  
