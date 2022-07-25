import string
import numpy as np
from keras.utils import to_categorical
from tensorflow import keras
from Data.image import load_image
from Data import params as p

model = keras.models.load_model(r"C:\Users\r0583\Downloads\keras_cifar10_trained_model (1).h5")

def predict(img:np.ndarray)->string:
    img = img.astype('float32')
    img /= 255
    image = img.reshape(-1, img.shape[0], img.shape[1], img.shape[2])
    label = model.predict(image)[0]
    # for i in range(len(label)):
    #     if (label[i] == 1):
    #         return(p.labels[i])


def main():
    data = np.load(p.binary_file_path)
    x_test = data['x_test']
    y_test = data['y_test']
    y_test = np.reshape(y_test, (-1, 1))
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test:np.ndarray = to_categorical(y_test)
    scores = model.evaluate(x_test, y_test, verbose=1)
    pred=model.predict(x_test)
