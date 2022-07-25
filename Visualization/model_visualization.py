from matplotlib import pyplot as plt
import numpy as np

from Model import model as m
from sklearn import metrics


import inspect

# confusion_matrix(model.data[])
print(m.model)

data=np.load(r"C:\Users\user1\Documents\bootcamp\Project\data.npz")
pred = m.model.predict(data['x_test'])

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
    """
    A function to annotate a heatmap.
    """
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = im.axes.text(j, i, format(data[i, j], fmt), horizontalalignment="center",
                                color="white" if data[i, j] > threshold else "black")
            texts.append(text)

    return texts


def confision_mat():
  labels=['airplane', 'automobile', 'bird', 'cat',
  'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
  'household furniture','large carnivores',
  'large man-made outdoor things',
  'people','trees']

  # Convert predictions classes to one hot vectors
  type(data['y_test'])
  print(pred.shape)
  Y_pred_classes = np.argmax(pred, axis=1)
  # Convert validation observations to one hot vectors
  Y_true = np.argmax(data['y_test'], axis=0)
  type(data['y_test'])
  # Errors are difference between predicted labels and true labels
  errors = (Y_pred_classes - Y_true != 0)

  Y_pred_classes_errors = Y_pred_classes[errors]
  Y_pred_errors = pred[errors]
  Y_true_errors = Y_true[errors]
  X_test_errors = data['x_test'[errors]]

  cm = metrics.confusion_matrix(Y_true, Y_pred_classes)
  thresh = cm.max() / 2.

  fig, ax = plt.subplots(figsize=(12,12))
  im, cbar = heatmap(cm, labels, labels, ax=ax,
                    cmap=plt.cm.Blues, cbarlabel="count of predictions")
  texts = annotate_heatmap(im, data=cm, threshold=thresh)

  fig.tight_layout()
  plt.show()
  return cm
confision_mat()
