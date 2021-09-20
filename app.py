from main_revisi2 import *
import cv2
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
import matplotlib.colors as colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
import time
from PyQt5.QtWidgets import *
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename
root = Tk()
root.withdraw()


# class MatplotlibWidget(QWidget):
#	def __init__(self, parent=None):
#		super(MatplotlibWidget, self).__init__(parent)
#		self.figure = Figure(figsize=(12, 12))
#		self.canvas = FigureCanvasQTAgg(self.figure)
#		self.axis1 = self.figure.add_subplot(1,1,1)
#		self.layoutvertical = QVBoxLayout(self)
#		self.layoutvertical.addWidget(self.canvas)


class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        self.timer1 = QTimer()

        # set timer timeout callback function
        self.timer.timeout.connect(self.openCamera)
        self.timer1.timeout.connect(self.detectFaces)

        # set control_bt callback clicked  function
        self.ui.camButton.clicked.connect(self.cameraTimer)
        self.ui.monButton.clicked.connect(self.monitorTimer)
        self.ui.haarButton.clicked.connect(self.haarcascade)
        self.ui.processButton .clicked.connect(self.procesfunc)
        self.ui.modButton.clicked.connect(self.modfunc)
        self.ui.startButton.clicked.connect(self.mainPage)

    def mainPage(self):
        self.ui.tabWidget.setCurrentIndex(1)

    def haarcascade(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        detection_model, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "(*.xml)", options=options)
        self.face_cascade = cv2.CascadeClassifier(detection_model)

    def modfunc(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        emotion_model, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "(*.hdf5 *.h5)", options=options)
        self.model = load_model(emotion_model, compile=False)

    # open camera

    def openCamera(self):
        ret, frame = self.cap.read()
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = self.frame.shape
        step = channel * width
        qImg = QImage(self.frame.data, width, height,
                      step, QImage.Format_RGB888)
        self.ui.camLabel.setPixmap(QPixmap.fromImage(qImg))
        self.ui.camLabel.setAlignment(
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    #pop-up prepocessing

    def procesfunc(self):
        scaling_factor = 1
        frame = cv2.resize(self.frame, None, fx=scaling_factor,
                           fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(face_rects) > 0:
            # Hanya bekerja dengan wajah utama dalam gambar (wajah dengan area terluas)
            face = sorted(face_rects, reverse=True, key=lambda x: (
                                                                          x[2] - x[0]) * (x[3] - x[1]))[0]
            (x, y, h, w) = face
            # Pisahkan wajah yang baru ditemukan dan ubah ukurannya menjadi ukuran 48x48 untuk mempersiapkan penyertaan di Jaringan Neural
            roi_gray = gray[y:y + w, x:x + h]
            roi_resize = cv2.resize(roi_gray, (48, 48))

        cv2.imshow('greyscale', roi_gray)
        cv2.imshow('resize', roi_resize)

    # detect face

    def detectFaces(self):
        # resize frame image
        scaling_factor = 1
        frame = cv2.resize(self.frame, None, fx=scaling_factor,
                           fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # Beralih dari warna ke abu-abu menggunakan OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect rect faces
        face_rects = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        canvas = np.zeros((250, 300, 3), dtype="uint8")

        # Mengkategorikan emosi berdasarkan 7 emosi dasar
        EMOTIONS = ['Marah', 'Jijik', 'Takut',
                    'Senang', 'Sedih', 'Terkejut', 'Biasa']
        EMOTIONX = ['Frustasi', 'Tidak Tertarik', 'Tidak Tertarik',
                    'Tertarik', 'Frustasi', 'Terkejut', 'Biasa']

        if len(face_rects) > 0:
            # Hanya bekerja dengan wajah utama dalam gambar (wajah dengan area terluas)
            face = sorted(face_rects, reverse=True, key=lambda x: (
                x[2] - x[0]) * (x[3] - x[1]))[0]
            (x, y, h, w) = face
            # Pisahkan wajah yang baru ditemukan dan ubah ukurannya menjadi ukuran 48x48 untuk mempersiapkan penyertaan di Jaringan Neural
            roi_gray = gray[y:y+w, x:x+h]
            roi = cv2.resize(roi_gray, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Lakukan prediksi emosional
            preds = self.model.predict(roi)[0]
            label = EMOTIONX[preds.argmax()]
            emotion_probability = np.max(preds)

            # Lampirkan label emosi yang dapat diprediksi pada gambar
            cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 191, 255), 2)
            
            # Menunjukkan emosi pada label 2 emosi dominan
            self.ui.label_2.setText(label)

            #Kotak Probabilitas yang terdekti oleh sistem
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 0)
                canvas = cv2.rectangle(
                    canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 191, 255), 1)

        heightC, widthC, channelC = canvas.shape
        stepC = channelC * widthC
        qImgC = QImage(canvas.data, widthC, heightC,
                       stepC, QImage.Format_RGB666)
        self.ui.probLabel.setPixmap(QPixmap.fromImage(qImgC))
        self.ui.probLabel.setAlignment(
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # get frame infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from RGB frame
        qImg = QImage(frame.data, width, height, step, QImage.Format_BGR888)
        # show frame in img_label
        self.ui.camLabel.setPixmap(QPixmap.fromImage(qImg))
        self.ui.camLabel.setAlignment(
            QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def cameraTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.camButton.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.camButton.setText("Rekam")

    def monitorTimer(self):
        # if timer is stopped
        if not self.timer1.isActive():
            # create video capture
            self.cap1 = cv2.VideoCapture(0)
            # start timer
            self.timer1.start(20)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
