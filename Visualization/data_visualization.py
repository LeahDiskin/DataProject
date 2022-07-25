import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Utils.params as p
from Data import datasets as d

df=pd.read_csv(p.csv_path)

#display num of sempals for each class using bar
def bar_samples_per_class(df):

    classes=df[p.labels_col_name_df].unique()
    sempals_per_class=[]
    for i in classes:
        sempals_per_class.append(df[df[p.labels_col_name_df]==i].count()[0])
    x = np.array(classes)
    y = np.array(sempals_per_class)
    plt.xticks(x)
    plt.tick_params(axis='x', colors='red', direction='out', length=13, width=3)
    plt.bar(x,y)
    plt.show()

#display the split to train/test/validation using multy bar
def display_split_train_test_validation(df):

    data = np.load(r"C:\Users\IMOE001\Downloads\data.npz")
    y_train =pd.DataFrame(data['y_train'])
    y_validation = pd.DataFrame(data['y_validation'])
    y_test = pd.DataFrame(data['y_test'])

    # train, validation, test = split_data(p.csv_path)
    labels = df['labels'].unique()



    fig = plt.figure()
    ax = fig.add_subplot(111)
    # matplotlib 3.0 you have to use align
    ax.bar(labels - p.width, y_train.groupby(0)[0].count(), p.width, color='b',tick_label=labels, label='-Ymin', align='edge')
    ax.bar(labels , y_validation.groupby(0)[0].count(), p.width, color='r',tick_label=labels, label='Ymax',
           align='edge')
    ax.bar(labels + p.width ,y_test.groupby(0)[0].count(), p.width, color='g',tick_label=labels, label='-Ymin',
           align='edge')

    ax.set_xlabel('train/validation/test')

    plt.show()

#display some images for each class
def display_data_example(df):

    labels = df[p.labels_col_name_df].unique()
    fig = plt.figure(figsize=(100, 400),constrained_layout=True)
    # setting values to rows and column variables
    rows = 15
    columns = 8
    pos = 1
    for i in labels:
        ax = fig.add_subplot(rows, columns, pos)
        ax.set_title(p.labels[int(i)],loc="left",size="small")
        plt.axis('off')
        for j in df[df[p.labels_col_name_df]==i].head(columns)[p.path_col_name]:

             fig.add_subplot(rows, columns, pos)

             image = plt.imread(j)
             # showing image
             plt.imshow(image)
             plt.axis('off')
             plt.subplot_tool()
             plt.subplots_adjust(wspace=0, hspace=0.1)

             # plt.subplots_adjust(left=0.1,
             #                     bottom=0.1,
             #                     right=0.9,
             #                     top=0.9,
             #                     wspace=0.4,
             #                     hspace=2)
             pos+=1
    plt.show()


def main():
    bar_samples_per_class(df)
    display_split_train_test_validation(df)
    display_data_example(df)

if __name__=="__main__":
    main()