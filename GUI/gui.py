import os.path

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QDir
from PyQt5.QtWidgets import QComboBox, QMainWindow, QApplication, QWidget, QVBoxLayout, QFileDialog, QPushButton, QLabel
from PyQt5.QtGui import QIcon, QPixmap
import sys
from Utils import params as p
from Data.image import image_datails_to_csv, load_image, image_to_cifar10_format, image_to_square
from Model import model
from pathlib import Path

class PhotoLabel(QLabel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
        QLabel {
            border: 4px dashed #aaa;
        }''')


    def setPixmap(self, *args, **kwargs):
        super().setPixmap(*args, **kwargs)
        self.setStyleSheet('''
        QLabel {
            border: none;
        }''')

class MainWindow(QMainWindow,QWidget):
    def __init__(self):
        super().__init__()
        labels=p.labels
        self.select_label = QComboBox(self)
        self.select_label.addItems(labels)
        layout = QVBoxLayout()
        layout.addWidget(self.select_label)
        self.filename=""

        #image
        self.photo = PhotoLabel()
        browse_btn = QPushButton('Browse')
        browse_btn.clicked.connect(self.open_image)
        # grid = QGridLayout(self)
        layout.addWidget(browse_btn)
        layout.addWidget(self.photo)
        self.setAcceptDrops(True)
        self.resize(300, 200)

        # crop
        # self.photo = PhotoLabel()
        browse_btn = QPushButton('crop')
        browse_btn.clicked.connect(self.open_image)
        # grid = QGridLayout(self)
        layout.addWidget(browse_btn)
        layout.addWidget(self.photo)
        self.setAcceptDrops(True)
        self.resize(300, 200)

        #save
        save_btn = QPushButton('Save')
        save_btn.clicked.connect(self.save_image)
        layout.addWidget(save_btn)

        # get_prediction
        prediction = QPushButton('Prediction')
        prediction.clicked.connect(self.get_prediction)
        layout.addWidget(prediction)
        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

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
        if not self.filename:
            self.filename, _ = QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(),
                                                      'Images (*.png *.jpg)')
            if not self.filename:
                return
        image_pixmap: QPixmap =self.photo.setPixmap(QPixmap(self.filename))

            # self.pixmap().copy(filename)



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

        print(model.predict(img))


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec_()

# load all images from source folder , convert to cifar10 format and save in dest folder
# def images_to_cifar10_format(source_path, dest_path):
#     for dirname, dirnames, filenames in os.walk(source_path):
#         for filename in filenames:
#             if filename.endswith('.JPG') or filename.endswith('.jpg') or filename.endswith('.pmg'):
#                 img = Image.open(os.path.join(dirname, filename))
#                 img = image_to_cifar10_format(img)
#                 img.save(dest_path + "/" + filename)

# import sys
#
# from PySide2.QtCore import QRect, QSize, QPoint
#
# from PySide2.QtWidgets import QLabel, QRubberBand, QApplication, QWidget
#
# from PySide2.QtGui import QPixmap, QMouseEvent
#
#
# class QExampleLabel(QLabel):
#
#     def __init__(self, parent_widget: QWidget = None):
#         super(QExampleLabel, self).__init__(parent_widget)
#
#         self.origin_point: QPoint = None
#
#         self.current_rubber_band: QRubberBand = None
#
#         self.init_ui()
#
#     def init_ui(self):
#         self.setPixmap(QPixmap(r"C:\Users\r0583\Documents\Bootcamp\project\new_images\depositphotos_3054837-stock-photo-truck-with-freight.jpg"))
#
#     def mousePressEvent(self, mouse_event: QMouseEvent):
#         self.origin_point = mouse_event.pos()
#
#         self.current_rubber_band = QRubberBand(QRubberBand.Rectangle, self)
#
#         self.current_rubber_band.setGeometry(QRect(self.origin_point, QSize()))
#
#         self.current_rubber_band.show()
#
#     def mouseMoveEvent(self, mouse_event: QMouseEvent):
#         self.current_rubber_band.setGeometry(QRect(self.origin_point, mouse_event.pos()).normalized())
#
#     def mouseReleaseEvent(self, mouse_event: QMouseEvent):
#         self.current_rubber_band.hide()
#
#         current_rect: QRect = self.current_rubber_band.geometry()
#
#         self.current_rubber_band.deleteLater()
#
#         crop_pixmap: QPixmap = self.pixmap().copy(current_rect)
#
#
#         crop_pixmap.save(r"C:\Users\r0583\Documents\Bootcamp\project\crop.png")
#
#
# if __name__ == '__main__':
#     myQApplication = QApplication(sys.argv)
#
#     myQExampleLabel = QExampleLabel()
#
#     myQExampleLabel.show()
#
#     sys.exit(myQApplication.exec_())

# import sys
#
# from PySide2.QtWidgets import (QApplication, QComboBox, QDialog,
#
#                                QDialogButtonBox, QGridLayout, QGroupBox,
#
#                                QFormLayout, QHBoxLayout, QLabel, QLineEdit,
#
#                                QMenu, QMenuBar, QPushButton, QSpinBox,
#
#                                QTextEdit, QVBoxLayout)
#
#
# class Dialog(QDialog):
#     num_grid_rows = 3
#
#     num_buttons = 4
#
#     def __init__(self):
#
#         super().__init__()
#
#         self._small_editor = None
#
#         self._file_menu = None
#
#         self._menu_bar = None
#
#         self._horizontal_group_box = None
#
#         self._grid_group_box = None
#
#         self._exit_action = None
#
#         self._form_group_box = None
#
#         self.create_menu()
#
#         self.create_horizontal_group_box()
#
#         self.create_grid_group_box()
#
#         self.create_form_group_box()
#
#         big_editor = QTextEdit()
#
#         big_editor.setPlainText("This widget takes up all the remaining space "
#
#                                 "in the top-level layout.")
#
#         button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
#
#         button_box.accepted.connect(self.accept)
#
#         button_box.rejected.connect(self.reject)
#
#         main_layout = QVBoxLayout()
#
#         main_layout.setMenuBar(self._menu_bar)
#
#         main_layout.addWidget(self._horizontal_group_box)
#
#         main_layout.addWidget(self._grid_group_box)
#
#         main_layout.addWidget(self._form_group_box)
#
#         main_layout.addWidget(big_editor)
#
#         main_layout.addWidget(button_box)
#
#         self.setLayout(main_layout)
#
#         self.setWindowTitle("Basic Layouts")
#
#     def create_menu(self):
#
#         self._menu_bar = QMenuBar()
#
#         self._file_menu = QMenu("&File", self)
#
#         self._exit_action = self._file_menu.addAction("E&xit")
#
#         self._menu_bar.addMenu(self._file_menu)
#
#         self._exit_action.triggered.connect(self.accept)
#
#     def create_horizontal_group_box(self):
#
#         self._horizontal_group_box = QGroupBox("Horizontal layout")
#
#         layout = QHBoxLayout()
#
#         for i in range(Dialog.num_buttons):
#             button = QPushButton(f"Button {i + 1}")
#
#             layout.addWidget(button)
#
#         self._horizontal_group_box.setLayout(layout)
#
#     def create_grid_group_box(self):
#
#         self._grid_group_box = QGroupBox("Grid layout")
#
#         layout = QGridLayout()
#
#         for i in range(Dialog.num_grid_rows):
#             label = QLabel(f"Line {i + 1}:")
#
#             line_edit = QLineEdit()
#
#             layout.addWidget(label, i + 1, 0)
#
#             layout.addWidget(line_edit, i + 1, 1)
#
#         self._small_editor = QTextEdit()
#
#         self._small_editor.setPlainText("This widget takes up about two thirds "
#
#                                         "of the grid layout.")
#
#         layout.addWidget(self._small_editor, 0, 2, 4, 1)
#
#         layout.setColumnStretch(1, 10)
#
#         layout.setColumnStretch(2, 20)
#
#         self._grid_group_box.setLayout(layout)
#
#     def create_form_group_box(self):
#
#         self._form_group_box = QGroupBox("Form layout")
#
#         layout = QFormLayout()
#
#         layout.addRow(QLabel("Line 1:"), QLineEdit())
#
#         layout.addRow(QLabel("Line 2, long text:"), QComboBox())
#
#         layout.addRow(QLabel("Line 3:"), QSpinBox())
#
#         self._form_group_box.setLayout(layout)
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#
#     dialog = Dialog()
#
#     sys.exit(dialog.exec())
