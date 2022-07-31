import os
import sys
import cv2
import numpy as np
from ctypes.wintypes import RGB
from PIL import Image, ImageOps
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QDir, QPoint, QRect, QSize, QFile
from PyQt5.QtWidgets import QComboBox, QMainWindow, QApplication, QWidget, QVBoxLayout, QFileDialog, QPushButton, QLabel, QHBoxLayout, QGridLayout, QRubberBand
from PyQt5.QtGui import QPixmap, QPalette, QMouseEvent
from Model import model
from Utils import params as p
from Data.image import image_datails_to_csv, load_image, image_to_cifar10_format, image_to_square
from pathlib import Path
from PIL import Image, ImageQt
from datetime import datetime
from playsound import playsound
from PyQt5.QtWidgets import QMessageBox

playsound_path=r"C://Users//user1//Downloads//camera-shutter-click-03.WAV"


class PhotoLabel(QLabel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(400,400)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
        QLabel {
            border: 4px dashed #aaa;
        }''')
        #path for saving images
        self.filename = ""

        #declare sub laout
        self.layout = QVBoxLayout()

        # lable predict
        self.label_predict = QLabel("", self)
        self.layout.addWidget(self.label_predict)


    def setPixmap(self, *args, **kwargs):
        super().setPixmap(*args, **kwargs)
        self.setGeometry(QtCore.QRect(10, 50, 400, 400))
        self.setStyleSheet('''
        QLabel {
            border: none;
        }''')

    def showImage(self, image_path: str):
        #reaset the predict label
        self.label_predict.setText("")
        #image to squre
        image = Image.open(image_path)
        h = image.height
        w = image.width
        if w > h:
            border = (0, (w - h) // 2, 0, (w - h) // 2)
            image = ImageOps.expand(image, border=border, fill=RGB(252, 252, 252))
        elif h > w:
            border = ((h - w) // 2, 0, (h - w) // 2, 0)
            image = ImageOps.expand(image, border=border, fill=RGB(252, 252, 252))
        image = image.resize((400, 400))
        image.save("output.png")
        #display the image on the gui
        image_pixmap: QPixmap = self.setPixmap(QPixmap("output.png"))


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

        crop_pixmap: QPixmap = self.pixmap().copy(current_rect)

        path_crop=r"C:\Users\IMOE001\Desktop\studied\aplied_material\project\crop"

        #give the image uniqe path
        str = datetime.now().strftime("%d-%m-%Y %H;%M;%S")
        self.filename = os.path.join(path_crop, f"crop_{str}.png")

        #save the crop image
        crop_pixmap.save(self.filename)

        #display the crop image
        self.showImage(self.filename)



class MainWindow(QMainWindow,QWidget):

    def __init__(self, parent_widget: QWidget = None):

        super(MainWindow, self).__init__(parent_widget)

        self.image=PhotoLabel()

        self.origin_point: QPoint = None

        self.current_rubber_band: QRubberBand = None

        # create main and sub layouts
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

        #image
        self.photo = PhotoLabel()
        self.photo.setGeometry(QtCore.QRect(10, 50, 400, 400))
        layout.addLayout(self.photo.layout)
        layout.addWidget(self.photo, alignment=Qt.AlignCenter)

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

        self.msgBox = QMessageBox()

        #declare container
        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)
        self.setAcceptDrops(True)
        self.resize(300, 200)

        self.init_ui()



    def init_ui(self):
        self.image.setPixmap(QPixmap(self.photo.filename))

    def current_text_changed(self, s):
        print("Current text: ", s)

    # this function loads an image
    def open_image(self):
        self.photo.filename, _ = QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(),
                                                      'Images (*.png *.jpg)')
        if not self.photo.filename:
                return
        self.photo.showImage(self.photo.filename)


    def save_image(self,event):

        # load an image
        img:np.ndarray=load_image(self.photo.filename)
        img: Image = Image.fromarray(img)

        # image to cifar10 format: cut and resize
        img=image_to_cifar10_format(img)

        # image details to csv
        label=p.labels.index(self.select_label.currentText())
        image_name=Path(self.photo.filename).name
        image_path=fr"{p.new_images_folder_path}\{image_name}"
        csv_path=p.new_images_csv_path
        image_datails_to_csv(label,image_name,image_path,csv_path)
        self.msgBox.about(self,"saved successfully")

    def get_prediction(self):
        # load an image
        img: np.ndarray = load_image(self.photo.filename)
        img: Image = Image.fromarray(img)

        # image to cifar10 format: cut and resize
        img:np.ndarray=image_to_cifar10_format(img)

        #diaplay the prediction
        # self.photo.label_predict.setText(model.predict(img))

        pred=model.predict(img)
        res=pred[0]+'     '+str(pred[1])+'%'+'\n'+pred[2]

        self.msgBox.about(self,"prediction",res)

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
                # playsound(playsound_path)
                return_value, image = vc.read()
                break

        cv2.destroyWindow("camera")
        return image

    def open_camera(self):
        save = r"C:\Users\IMOE001\Desktop\studied\aplied_material\project\new images"

        image=self.new_window_camera()
        img = Image.fromarray(image)

        #give image uniqe path
        str=datetime.now().strftime("%d-%m-%Y %H;%M;%S")
        self.photo.filename=os.path.join(save, f"camera_{str}.png")

        #save image
        img.save(self.photo.filename)
        # display image
        self.photo.showImage(self.photo.filename)


def setColor(app):
    app.setStyle("Fusion")
    qp = QPalette()
    qp.setColor(QPalette.ButtonText, Qt.black)
    qp.setColor(QPalette.Window, Qt.white)
    qp.setColor(QPalette.Button, Qt.gray)
    app.setPalette(qp)
    return app


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    app=setColor(app)
    w.show()
    app.exec_()


if __name__=="__main__":
    main()