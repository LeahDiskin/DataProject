import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Data.datasets import split_data,csv_path
import json
from pathlib import Path

csv_path=r"C:\Users\IMOE001\Desktop\studied\aplied_material\project\cifar.csv"
df=pd.read_csv(csv_path)

#display num of sempals for each class using bar
def bar_samples_per_class(df):

    classes=df['labels'].unique()
    sempals_per_class=[]
    for i in classes:
        sempals_per_class.append(df[df['labels']==i].count()[0])
    x = np.array(classes)
    y = np.array(sempals_per_class)
    plt.xticks(x)
    plt.tick_params(axis='x', colors='red', direction='out', length=13, width=3)
    plt.bar(x,y)
    plt.show()

#display the split to train/test/validation using multy bar
def display_split_train_test_validation(df):

    train, validation, test = split_data(csv_path)
    labels = df['labels'].unique()
    # Calculate optimal width
    width = np.min(np.diff(labels)) / 20

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # matplotlib 3.0 you have to use align
    ax.bar(labels - width, train.groupby("labels")["image_name"].count(), width, color='b',tick_label=labels, label='-Ymin', align='edge')
    ax.bar(labels - width * 2, validation.groupby("labels")["image_name"].count(), width, color='r',tick_label=labels, label='Ymax',
           align='edge')
    ax.bar(labels - width * 3, test.groupby("labels")["image_name"].count(), width, color='g',tick_label=labels, label='-Ymin',
           align='edge')

    ax.set_xlabel('train/validation/test')

    plt.show()

#display some images for each class
def display_data_example(df):
    labels = df['labels'].unique()
    for i in labels:
        for j in df[df["labels"]==i].head(5):
             image = plt.imread(j["image_path"])
             plt.imshow(i,image)



bar_samples_per_class(df)
display_split_train_test_validation(df)
display_data_example(df)