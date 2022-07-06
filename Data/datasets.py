import numpy as np
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import csv
import os
from typing import Dict

images, labels, labels_name, images_names, batch, images_path = [], [], [], [], [], []
cifar100_images, cifar10_images = [], []

# load function: receive path and load the file into dict
def load(file:str)->Dict:
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

# insert the meta-data into csv: receive df and insert to csv
def data_to_csv(df_of_all_meta_data):
    df_of_all_meta_data.to_csv(csv_path)

# organize the data: data to dataframe
def organize_cifar10(data)->pd.DataFrame:
        return pd.DataFrame({labels_col_name:data[b'labels'] , images_col_name: data[b'filenames'], dataset_col_name: "cifar10" ,file_in_dataset_col_name: data[b'batch_label']})

def create_df()->pd.DataFrame:
    return pd.DataFrame({labels_col_name:[], images_col_name: [], dataset_col_name: [],file_in_dataset_col_name: []} )#function creat df

#############
#  cifar10  #
#############
#
def cifar10()->pd.DataFrame:
    cifar10_meta_data=create_df()
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
    #I would like to change this two lines
    train_df = pd.DataFrame({"labels": train_data[b'coarse_labels'],"image_name": train_data[b'filenames'], "dataset":"cifar100", "batch/train/test": "train"})
    test_df = pd.DataFrame({"labels": test_data[b'coarse_labels'],"image_name": test_data[b'filenames'], "dataset":"cifar100", "batch/train/test": "test"})
    cifar100_df = pd.concat([train_df, train_df])
    # meta data of only chosen classes
    cifar100_meta_data = cifar100_df[cifar100_df.labels.isin(chosen_classes_from_cifar100)]
    # extract images from chosen classes
    delete_images(cifar100_meta_data)
    # cifar100_meta_data["dataset"]="cifar100"
    return cifar100_meta_data

def images_to_folder(images,images_names):
        for i in range(0, len(images)):
            img = Image.fromarray(images[i])
            img.save(f"{image_folder_path}/{images_names[i]}")

##############
#  cifar100  #
##############
def cifar100():

    # load data from cifar100
    meta_data=load(fr"{cifar_100_path}\meta")
    train_data=load(fr"{cifar_100_path}\train")
    test_data=load(fr"{cifar_100_path}\test")

    # create array of images
    images_array(train_data, cifar100_images)
    images_array(test_data, cifar100_images)

    # extract and organize data from chosen classes
    cifar100_meta_data=organize_cifar100_chosen_data(train_data,test_data)
    return cifar100_meta_data

def split_data(path):
    data_from_csv = pd.read_csv(path)
    train, validation, test = np.split(data_from_csv.sample(frac=1, random_state=42),[int(.6*len(data_from_csv)), int(.8*len(data_from_csv))])
    return train,validation,test




def main():

    cifar10_meta_data:pd.DataFrame=cifar10()
    cifar100_meta_data:pd.DataFrame=cifar100()

    #concat dataframes
    all_meta_data:pd.DataFrame=pd.concat([cifar10_meta_data,cifar100_meta_data])
    images:np.array=cifar100_images+cifar10_images

    #images_to_folder(images,df_of_all_meta_data[b'filenames'])

    if (os.path.exists(csv_path) and os.path.isfile(csv_path)):
        os.remove(csv_path)
        print("file deleted")

    data_to_csv(all_meta_data)

    #split to train, test, validation
    train, validation, test=split_data(csv_path)
if __name__=="__main__":
#insert to json
    cifar10_path = Path(r"C:\Users\user1\Documents\bootcamp\Project\cifar-10-batches-py")
    cifar_100_path = Path(r"C:\Users\user1\Documents\bootcamp\Project\cifar-100-python")
    image_folder_path = Path(r"C:\Users\user1\Documents\bootcamp\Project\project\images")
    csv_path = Path(r"C:\Users\user1\Documents\bootcamp\Project\cifar.csv")

    chosen_classes_from_cifar100 = [6, 8, 9, 14, 17] #dict
    labels_col_name='labels'
    images_col_name='image_name'
    dataset_col_name='dataset'
    file_in_dataset_col_name='batch/train/test'

    main()