import os

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QDir, QPoint, QRect, QSize, QFile
from PyQt5.QtWidgets import QComboBox, QMainWindow, QApplication, QWidget, QVBoxLayout, QFileDialog, QPushButton, \
    QLabel, QHBoxLayout, QGridLayout, QRubberBand
from PyQt5.QtGui import QPixmap, QPalette, QMouseEvent
import sys

from Model import model
from Utils import params as p
from Data.image import image_datails_to_csv, load_image, image_to_cifar10_format, image_to_square
#from Model import model
from pathlib import Path

from PIL import Image, ImageQt
from datetime import datetime

import cv2
from playsound import playsound

playsound_path=Path(r"C://Users//user1//Downloads//camera-shutter-click-03.wav")



# from PySide2.QtGui import QPixmap, QMouseEvent, QPalette
class PhotoLabel(QLabel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignCenter)
        # self.setFixedSize(500,500)

        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
        QLabel {
            border: 4px dashed #aaa;
        }''')


    def setPixmap(self, *args, **kwargs):
        super().setPixmap(*args, **kwargs)
        self.setGeometry(QtCore.QRect(50, 150, 400, 400))
        self.setStyleSheet('''
        QLabel {
            border: none;
        }''')

class MainWindow(QMainWindow,QWidget):

    def __init__(self, parent_widget: QWidget = None):

        super(MainWindow, self).__init__(parent_widget)

        self.image=PhotoLabel()

        self.origin_point: QPoint = None

        self.current_rubber_band: QRubberBand = None



        # self.center_image_crop=(0,0)

        self.filename = ""
        self.init_ui()

        # create main and sub layout
        layout = QVBoxLayout()
        layout1 = QHBoxLayout(self)
        layout2 = QHBoxLayout()

        #select label
        self.select_label = QComboBox(self)
        self.select_label.addItems(p.labels)
        layout1.addWidget(self.select_label)

        #browse
        self.browse_btn = QPushButton('Browse')
        self.browse_btn.clicked.connect(self.open_image)
        layout1.addWidget(self.browse_btn)

        #set the sub layout in the main one
        layout.addLayout(layout1)

        #lable predict
        self.label_predict = QLabel("", self)
        layout.addWidget(self.label_predict)

        #image
        self.photo = PhotoLabel()
        self.photo.setGeometry(QtCore.QRect(50, 150, 400, 400))
        layout.addWidget(self.photo)

        #save
        self.save_btn = QPushButton('Save')
        self.save_btn.clicked.connect(self.save_image)
        layout2.addWidget(self.save_btn)

        #prediction
        self.prediction = QPushButton('Prediction')
        self.prediction.clicked.connect(self.get_prediction)
        layout2.addWidget(self.prediction)

        # open camera
        camera_btn = QPushButton('open camera')
        camera_btn.clicked.connect(self.open_camera)
        layout.addWidget(camera_btn)

        #set the sub layout in the main one
        layout.addLayout(layout2)

        #declare container
        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)
        self.setAcceptDrops(True)
        self.resize(300, 200)

        ##try anther type of layout
        # grid = QGridLayout(self)
        # grid.addWidget(self.select_label, 0, 1)
        # grid.addWidget(self.browse_btn, 0, 3)
        # grid.addWidget(self.photo, 2, 0)

    def init_ui(self):
        # self.image.setPixmap(QPixmap('input.png'))
        self.image.setPixmap(QPixmap(self.filename))

    def current_text_changed(self, s):
        print("Current text: ", s)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            filename = event.mimeData().urls()[0].toLocalFile()
            event.accept()
            self.open_image(filename)
        else:
            event.ignore()

    # this function loads an image
    def open_image(self):
        # if not self.filename:
        self.filename, _ = QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(),
                                                      'Images (*.png *.jpg)')
            # if not self.filename:
            #     return
        image_pixmap: QPixmap =self.photo.setPixmap(QPixmap(self.filename))




    def save_image(self,event):

        # load an image
        img:np.ndarray=load_image(self.filename)
        img: Image = Image.fromarray(img)

        # image to cifar10 format: cut and resize
        img=image_to_cifar10_format(img)

        # image details to csv
        label=p.labels.index(self.select_label.currentText())
        image_name=Path(self.filename).name
        image_path=fr"{p.new_images_folder_path}\{image_name}"
        csv_path=p.new_images_csv_path
        image_datails_to_csv(label,image_name,image_path,csv_path)

    def get_prediction(self):
        # load an image
        img: np.ndarray = load_image(self.filename)
        img: Image = Image.fromarray(img)

        # image to cifar10 format: cut and resize
        img:np.ndarray=image_to_cifar10_format(img)

        #diaplay the prediction
        self.label_predict.setText(model.predict(img))


    def mousePressEvent(self, mouse_event: QMouseEvent):
        self.origin_point = mouse_event.pos()

        self.current_rubber_band = QRubberBand(QRubberBand.Rectangle, self)

        self.current_rubber_band.setGeometry(QRect(self.origin_point, QSize()))

        self.current_rubber_band.show()

    def mouseMoveEvent(self, mouse_event: QMouseEvent):
        self.current_rubber_band.setGeometry(QRect(self.origin_point, mouse_event.pos()).normalized())

    def mouseReleaseEvent(self, mouse_event: QMouseEvent):
        self.current_rubber_band.hide()

        current_rect: QRect = self.current_rubber_band.geometry()

        self.current_rubber_band.deleteLater()
        # crop_image=PhotoLabel()

        crop_pixmap: QPixmap = self.photo.pixmap().copy(current_rect)
        # self.filename, _ = QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(),
        #       'Images (*.png *.jpg)')
        path_crop=r'C:\Users\user1\Documents\bootcamp\Project\new_images'
        str = datetime.now().strftime("%d-%m-%Y %H;%M;%S")
        self.filename = os.path.join(path_crop, f"crop_{str}.png")
        # crop_pixmap.save(r"C://Users//IMOE001//Desktop//studied//aplied_material//project//new images//img.png","PNG")
        crop_pixmap.save(self.filename)
        self.photo.setPixmap(QPixmap(self.filename))


    # def QPixmapToArray(self,pixmap):
    #     ## Get the size of the current pixmap
    #     size = pixmap.size()
    #     h = size.width()
    #     w = size.height()
    #
    #     ## Get the QImage Item and convert it to a byte string
    #     qimg = pixmap.toImage()
    #     byte_str = qimg.bits().tobytes()
    #
    #     ## Using the np.frombuffer function to convert the byte string into an np array
    #     img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w, h, 4))
    #
    #     return img

    def new_window_camera(self):
        cv2.namedWindow("camera")
        vc = cv2.VideoCapture(0)

        if vc.isOpened():  # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
        while rval:
            cv2.imshow("camera", frame)
            rval, frame = vc.read()

            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                playsound(playsound_path)
                return_value, image = vc.read()
                break

        vc.release()
        cv2.destroyWindow("camera")
        return image

    def open_camera(self):
        save = r"C:/Users/user1/Documents/bootcamp/Project/bonus/from_webcamera/"

        image=self.new_window_camera()

        img = Image.fromarray(image)
        str=datetime.now().strftime("%d-%m-%Y %H;%M;%S")
        self.filename=os.path.join(save, f"camera_{str}.png")
        img.save(self.filename)
        self.photo.setPixmap(QPixmap(self.filename))








def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    app.setStyle("Fusion")
    qp = QPalette()
    qp.setColor(QPalette.ButtonText, Qt.black)
    qp.setColor(QPalette.Window, Qt.white)
    qp.setColor(QPalette.Button, Qt.red)
    app.setPalette(qp)
    w.show()
    app.exec_()

if __name__=="__main__":
    main()