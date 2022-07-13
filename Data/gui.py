from PyQt5.QtCore import Qt, QDir
from PyQt5.QtWidgets import QComboBox, QMainWindow, QApplication, QWidget, QVBoxLayout, QFileDialog, QPushButton, \
     QLabel
from PyQt5.QtGui import QIcon, QPixmap
import sys
import params as p
from Data.image import image_to_cifar10_format

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





        #image
        self.photo = PhotoLabel()
        browse_btn = QPushButton('Browse')
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
    def open_image(self, filename=None):
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(),
                                                      'Images (*.png *.jpg)')
            if not filename:
                return
        self.photo.setPixmap(QPixmap(filename))


    def save_image(self,event):
        # img=
        # image.image_to_cifar10_format()
        print("save")
        print(self.photo.toPlainText())
        #image_to_cifar10_format()
        #image_datails_to_csv()
        label=self.select_label.currentText()
        print(label)
    def get_prediction(self):
        print("prediction")

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