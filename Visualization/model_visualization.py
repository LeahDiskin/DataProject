import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import Utils.params as p
def present_wrong_predicted_labels(pred_labels:np.ndarray,true_labels:np.ndarray,images:np.ndarray):
    counter=0
    ind=0
    fig = plt.figure(figsize=(12, 12))
    rows = 4
    columns = 3
    while counter<12:
        true_label=true_labels[ind].argmax()
        pred_label=pred_labels[ind].argmax()
        if true_label!=pred_label:
            true_label_name=p.labels[true_label]
            pred_label_name=p.labels[pred_label]
        img = Image.fromarray(images[ind])
        fig.add_subplot(rows, columns, counter+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"\ntrue: {true_label_name}, predicted: {pred_label_name}")
        counter+=1
        ind+=1
    plt.show()
from PIL import Image
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from Model import model as m
from Data import params as p
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# this function print the model accuracy and loss
def results(scores):
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

# this function shows the model loss function and accuracy
def plotmodelhistory(history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(history['accuracy'])
    axs[0].plot(history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'validate'], loc='upper left')
    # summarize history for loss
    axs[1].plot(history['loss'])
    axs[1].plot(history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'validate'], loc='upper left')
    plt.show()

# this function presents images where the model was wrong
def present_wrong_predicted_labels(pred_labels:np.ndarray,true_labels:np.ndarray,images:np.ndarray):
    counter=0
    ind=0
    fig = plt.figure(figsize=(12, 12))
    rows = 4
    columns = 3
    while counter<12:
        true_label=true_labels[ind].argmax()
        pred_label=pred_labels[ind].argmax()
        if true_label!=pred_label:
            true_label_name=p.labels[true_label]
            pred_label_name=p.labels[pred_label]
            img = Image.fromarray(images[ind])
            fig.add_subplot(rows, columns, counter+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"\ntrue: {true_label_name}, predicted: {pred_label_name}")
            counter+=1
        ind+=1
    plt.show()

# this two functions creates the heatmap for the confusion matrix
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    """
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    return im, cbar
def annotate_heatmap(im, data=None, fmt="d", threshold=None):
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = im.axes.text(j, i, format(data[i, j], fmt), horizontalalignment="center",
                                 color="black")
            texts.append(text)

    return texts
def confusion_mat(Y_true,Y_pred_classes):
    cm = confusion_matrix(Y_true, Y_pred_classes)
    thresh = cm.max() / 2.0
    fig, ax = plt.subplots(figsize=(10, 10))
    im, cbar = heatmap(cm, p.labels, p.labels, ax=ax,
                          cmap=plt.cm.Blues, cbarlabel="count of predictions")
    texts = annotate_heatmap(im, data=cm, threshold=thresh)
    fig.tight_layout()
    plt.show()

# this function shows the score in each prediction
def score(pred,errors):
    fig = plt.figure(figsize=(12, 12))
    label_score=np.amax(pred,axis=1)
    true_labels_score=label_score[(errors==False)]
    wrong_labels_score=label_score[(errors==True)]
    num_of_predictions=len(label_score)
    num_of_good_predictions=len(true_labels_score)
    num_of_wrong_predictions=len(wrong_labels_score)
    y1=[]
    y2=[]
    ind = np.arange(0,10)
    width=0.7
    for i in np.arange(0.0,1.0,0.1):
        y1.append(np.count_nonzero(np.logical_and(true_labels_score>=i, true_labels_score<(i+0.1))))
        y2.append(np.count_nonzero(np.logical_and(wrong_labels_score>=i, wrong_labels_score<(i+0.1))))
    bar1 = plt.bar(ind, y2, width, color="red")
    bar2 = plt.bar(ind, y1, width,bottom=y2, color="gray")
    sum = [x + y for (x, y) in zip(y1, y2)]
    max=np.amax(sum)
    plt.ylabel('number of labels got this score')
    plt.title('Scores')
    plt.xticks(ind, ("0.0-0.1","0.1-0.2","0.2-0.3","0.3-0.4","0.4-0.5","0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9","0.9-1.0"))
    plt.yticks(np.arange(0,max+1000,1000))
    plt.legend((bar2[0], bar1[0]), ('good_predicts', 'wrong_predicts'),loc='upper left')

    f_size = 8
    for i,rect in enumerate(bar1):
        height=sum[i]
        precent = round((height / num_of_predictions) * 100, 3)
        if(sum[i]!=0):
            wrong_precent=round((y2[i] / sum[i]) * 100, 3)
            plt.text(rect.get_x() + rect.get_width()/ 2.0, height+4*f_size, f"{precent}%", fontsize=10, fontweight='bold',
            color ='black',ha='center', va='bottom')
            plt.text(rect.get_x() + rect.get_width() / 2.0, 0.25, f"{wrong_precent}%", fontsize=f_size, fontweight='bold',
                     color='black', ha='center', va='bottom')

    plt.show()

#  this function colculates the accuracy for each cifar
def accuracy_for_two_cifars(pred_labels,true_labels):
    in_cifar10=(true_labels<10)
    cifar10_true_labels=true_labels[(in_cifar10)]
    cifar10_pred_labels=pred_labels[(in_cifar10)]
    cifar100_true_labels = true_labels[(in_cifar10==False)]
    cifar100_pred_labels = pred_labels[(in_cifar10==False)]
    print(f"accuracy of cifar10: {accuracy_score(cifar10_true_labels, cifar10_pred_labels)}")
    print(f"accuracy of cifar100: {accuracy_score(cifar100_true_labels, cifar100_pred_labels)}")
