import string
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from tensorflow import keras
from Data.image import load_image
from Data import params as p
from Visualization import model_visualization as vs
from sklearn.metrics import confusion_matrix, classification_report
import itertools
from pathlib import Path


# model = keras.models.load_model(r"C:\Users\r0583\Downloads\model_softmax")
model = keras.models.load_model(r"C:\Users\r0583\Downloads\model_cnn")
# model = keras.models.load_model(r"C:\Users\r0583\Downloads\model_fnn")
# model = keras.models.load_model(r"C:\Users\r0583\Downloads\model_resnet50_imagenet")
# model = keras.models.load_model(r"C:\Users\r0583\Downloads\model_vgg_imagenet")

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
    x_test_norm = x_test.astype('float32')
    x_test_norm /= 255
    y_test_encoded:np.ndarray = to_categorical(y_test)
    scores:list = model.evaluate(x_test_norm, y_test_encoded, verbose=1)
    pred=model.predict(x_test_norm)
    Y_pred_classes = np.argmax(pred, axis=1)
    Y_true = np.argmax(y_test_encoded, axis=1)
    errors = (Y_pred_classes - Y_true != 0)
    Y_pred_classes_errors = Y_pred_classes[errors]
    Y_pred_errors = pred[errors]
    Y_true_errors = Y_true[errors]
    X_test_errors = x_test[errors]

    # sequential_history = pd.read_csv(r"C:\Users\r0583\Downloads\history_fnn.csv")
    # sequential_history = pd.read_csv(r"C:\Users\r0583\Downloads\history_resnet50_imagenet.csv")
    # sequential_history = pd.read_csv(r"C:\Users\r0583\Downloads\history_vgg_imagenet.csv")
    sequential_history = pd.read_csv(r"C:\Users\r0583\Downloads\history_cnn.csv")

    # show loss function and accuracy
    vs.plotmodelhistory(sequential_history)

    # show accuracy and loss
    vs.results(scores)

    #present images where the model was wrong
    vs.present_wrong_predicted_labels(pred,y_test_encoded,x_test)

    # confusion matrix
    vs.confusion_mat(Y_true,Y_pred_classes)

    # show score
    vs.score(pred,errors)

    vs.accuracy_for_two_cifars(Y_pred_classes,Y_true)




    print()
if __name__ == "__main__":
    main()
