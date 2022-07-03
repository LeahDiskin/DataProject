 #C:\Users\user1\Documents\bootcamp\Project\cifar-10-batches-py\data_batch_ leah's path
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
#
# # loading data
# def load_data(path):
#     # print(path)
#     with open(path, 'rb') as fo:
#         data = pickle.load(fo, encoding='bytes')
#     return data
#
# # loading cifar10, returning a list length=5 each node contains one batch
# def load_cifar10():
#     all_data_seperating_to_batches = []
#     for i in range(0, num_of_batches):
#         all_data_seperating_to_batches.append(load_data(r'C:\Users\r0583\Documents\Bootcamp\project\cifar-10-batches-py\data_batch_' + str(i + 1)))
#     return all_data_seperating_to_batches
#
# creating a matrix that includes labels, image name and original size.


# def images_only(data,images_col):
#     for i in range (len(data[images_col])):
#         images.append(data[images_col][i])
#
# num_of_batches = 5
# all_labels=[]
# all_images_names=[]
# all_sizes_of_images=[]
# images=[]
# cifar10_seperating_to_batches=load_cifar10()
# for i in cifar10_seperating_to_batches:
#     organize_meta_data(i,b'filenames',b'labels','32*32')
#     images_only(i,b'data')
# data_to_csv(all_labels,all_images_names,all_sizes_of_images)
# images_to_folder()
import numpy as np
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import csv
import os
# import tensorflow as tf


# Path to the unzipped CIFAR data
cifar10_path = Path(r'C:\Users\r0583\Documents\Bootcamp\project\cifar-10-batches-py/')
image_folder_path= r"C:\Users\r0583\Documents\Bootcamp\project\images"
csv_path=r'C:/Users/r0583/Documents/Bootcamp/project/cifar.csv'
cifar_100_path=r'C:\Users\r0583\Documents\Bootcamp\project\cifar-100-python'
indexes_of_chosen_classes_from_cifar100=[6,8,9,14,17]

def images_to_folder(images):
    for i in range (0,len(images)):
        img = Image.fromarray(images[i])
        img.save(image_folder_path+'\image'+str(i)+'.jpg')

def data_to_csv(labels,images_names,sizes_of_images,image_path):
    dict_of_all_meta_data = {'label': labels, 'image name': images_names, 'origin_size': sizes_of_images,'image_path':image_path}
    df_of_all_meta_data = pd.DataFrame(dict_of_all_meta_data)
    df_of_all_meta_data.to_csv(csv_path)

# load function provided by the CIFAR hosts
def load(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

images, labels, images_names,images_path=[],[],[],[]
def organize_the_data(data):
    for i, flat_im in enumerate(data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024: (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(data[b"labels"][i])
        # Save the image name
        images_names.append(data[b'filenames'][i])
        # Save the image path
        images_path.append(image_folder_path + "\image" + str(i))

def cifar10():
    for batch in cifar10_path.glob("data_batch_*"):
        batch_data = load(batch)
        organize_the_data(batch_data)
    # images_to_folder(images)
    if (os.path.exists(csv_path) and os.path.isfile(csv_path)):
        os.remove(csv_path)
        print("file deleted")
    else:
        print("file not found")
    data_to_csv(labels,images_names,"32*32",images_path)

# def save_only_chosen_data(data):
#     def predicate(x, allowed_labels=tf.constant([0, 1, 2])):
#         label = x[b'label']
#         isallowed = tf.equal(allowed_labels, tf.cast(label, allowed_labels.dtype))
#         reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
#         return tf.greater(reduced, tf.constant(0.))
#
#     dataset = dataset.filter(predicate).batch(20)
#
#     for i, x in enumerate(tfds.as_numpy(dataset)):
#         print(x['label'])
#     for i in range (0,len(data)):
#         if(data[b'coarse_labels'][i] in indexes_of_chosen_classes_from_cifar100):



def cifar100():
    meta_data=load(cifar_100_path+r'\meta')
    train_data=load(cifar_100_path+r'\train')
    test_data=load(cifar_100_path+r'\test')
    cifar100_labels = train_data[b'coarse_labels']
    cifar100_image_name = train_data[b'filenames']
    # cifar100_images=train_data[b'data']
    cifar100_images=[]
    cifar100_labels.append(test_data[b'coarse_labels'])
    cifar100_image_name.append(test_data[b'filenames'])
    cifar100_images.append(test_data[b'data'])
    for i in train_data[b'data']:
        cifar100_images.append(np.dstack(i))
    for i in test_data[b'data']:
        cifar100_images.append(np.dstack(i))
    df = pd.DataFrame({"labels": cifar100_labels, "image_name": cifar100_image_name})
    df=df[df.labels.isin(indexes_of_chosen_classes_from_cifar100)]
    c=0
    for i in range (len(cifar100_images)):
        if i not in (df.index):
            cifar100_images.pop(i - c)
            c += 1

    # cifar100_images=cifar100_images[cifar100_images.isin(df.index)]
    print()
# cifar10()
cifar100()
print()
