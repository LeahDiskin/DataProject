import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv

# loading data- 5 batches
def load_data():
    all_data=[]
    num_of_batches = 5
    for i in range(0, num_of_batches):
        path2 = r'C:\Users\user1\Documents\bootcamp\Project\cifar-10-batches-py\data_batch_' + str(i + 1)
        print(path2)
        with open(path2, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        print("set ", i, "\n", data)
        all_data.append(data)
    return all_data

# creating
def seperating_images_and_labels(data):
    all_labels = []
    all_images = []
    for j in data[b'labels']:
        all_labels.append(j)
    for j in data[b'filenames']:
        all_images.append(j)

load_data()
#loading meta data

# file=r'C:\Users\user1\Documents\bootcamp\Project\cifar-10-batches-py\batches.meta'
# with open(file, 'rb') as fo:
#         data = pickle.load(fo, encoding='bytes')
# type(data)
# print(data)




#
# print(len(all_labels))
# print(len(all_images))
#
#
# with open("out.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(a)