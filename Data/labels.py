import numpy as np
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import csv
import os
from typing import Dict

# load function: receive path and load the file into dict
def load(file:str)->Dict:
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

# insert the meta-data into csv: receive df and insert to csv
def data_to_csv(df_of_all_meta_data):
    df_of_all_meta_data.to_csv(csv_path)

# organize the data:data to dataframe
def organize_cifar10(data):
        return pd.DataFrame({'labels':data[b'labels'] , 'image_name': data[b'filenames'], 'dataset': "cifar10" ,'batch/train/test': data[b'batch_label']})

#############
#  cifar10  #
#############
#
def cifar10()->pd.DataFrame:
    cifar10_meta_data=pd.DataFrame({'labels':[], 'image_name': [], 'dataset': [],'batch/train/test': []} )#function creat df

    for batch in cifar10_path.glob('data_batch_*'):#param
        # load the data
        batch_data = load(batch)
        # create array of images
        images_array(batch_data,cifar10_images)#לבדוק איפה Kלשמור תמונות
        # organize the meta data

        cifar10_meta_data=pd.concat([cifar10_meta_data,organize_cifar10(batch_data)])
        # cifar10_meta_data.append(organize_cifar10(batch_data))

    return cifar10_meta_data


def images_array(data,array):
    for i, flat_im in enumerate(data[b"data"]):
        im_channels = []
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024: (j + 1) * 1024].reshape((32, 32))
            )
        # Save the original image
        array.append(np.dstack((im_channels)))

# delete images that are not from the chosen classes
def delete_images(df):
    num_of_deleted = 0
    for i in range(len(cifar100_images)):
        if i not in (df.index):
            cifar100_images.pop(i - num_of_deleted)
            num_of_deleted += 1

def organize_cifar100_chosen_data(train_data,test_data):
    df1 = pd.DataFrame({"labels": train_data[b'coarse_labels'],"image_name": train_data[b'filenames'], "dataset":"cifar100", "batch/train/test": "train"})
    df2 = pd.DataFrame({"labels": test_data[b'coarse_labels'],"image_name": test_data[b'filenames'], "dataset":"cifar100", "batch/train/test": "test"})
    df = pd.concat([df1, df2])
    # meta data of only chosen classes
    cifar100_meta_data = df[df.labels.isin(chosen_classes_from_cifar100)]
    # extract images from chosen classes
    delete_images(df)
    # cifar100_meta_data["dataset"]="cifar100"
    return cifar100_meta_data

def images_to_folder(images,images_names):
        for i in range(0, len(images)):
            img = Image.fromarray(images[i])
            img.save(image_folder_path + "/" +images_names[i])

##############
#  cifar100  #
##############
def cifar100():
    # load meta data
    meta_data=load(str(cifar_100_path)+r'\meta')
    # load data from train data file and test data file
    train_data=load(str(cifar_100_path)+r'\train')
    test_data=load(str(cifar_100_path)+r'\test')

    # create array of images
    images_array(train_data, cifar100_images)
    images_array(test_data, cifar100_images)

    # extract and organize data from chosen classes
    cifar100_meta_data=organize_cifar100_chosen_data(train_data,test_data)
    return cifar100_meta_data


def main():

    cifar10_meta_data:pd.DataFrame=cifar10()  #return dataframes
    cifar100_meta_data:pd.DataFrame=cifar100()  #return dataframes

    #concat dataframes
    df_of_all_meta_data=pd.concat([cifar10_meta_data,cifar100_meta_data])
    images=cifar100_images+cifar10_images

    #images_to_folder(images,df_of_all_meta_data[b'filenames'])

    if (os.path.exists(csv_path) and os.path.isfile(csv_path)):
        os.remove(csv_path)
        print("file deleted")

    data_to_csv(df_of_all_meta_data)

if __name__=="__main__":

    cifar10_path = Path(r"C:\Users\user1\Documents\bootcamp\Project\cifar-10-batches-py")
    cifar_100_path = Path(r"C:\Users\user1\Documents\bootcamp\Project\cifar-100-python")
    image_folder_path = Path(r"C:\Users\user1\Documents\bootcamp\Project\project\images")
    csv_path = Path(r"C:\Users\user1\Documents\bootcamp\Project\cifar.csv")

    chosen_classes_from_cifar100 = [6, 8, 9, 14, 17]#dict
    images, labels, labels_name, images_names, batch, images_path = [], [], [], [], [], []
    cifar100_images, cifar10_images = [], []

    main()