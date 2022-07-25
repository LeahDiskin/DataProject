import pickle
import pandas as pd
import numpy as np
from PIL import Image
import os
from typing import Dict
from Data.image import load_image
from Utils import params as p

cifar100_images, cifar10_images = [], []
train_images, test_images = [], []

# this function loads data
def load(path:str)->Dict:
    with open(path, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

# this function creats an empty df
def create_df()->pd.DataFrame:
    return pd.DataFrame({p.labels_col_name_df:[], p.images_col_name_df: [], p.dataset_col_name_df: [],p.path_col_name_df:[]} ) #function creat df

# this function inserts data into df
def data_to_df(data:dict,labels_name,images_name,dataset,image_folder_path)->pd.DataFrame:
    df= pd.DataFrame({p.labels_col_name_df:data[labels_name] , p.images_col_name_df: data[images_name], p.dataset_col_name_df: dataset,p.path_col_name_df:image_folder_path })
    for i in df.index:
        df[p.path_col_name_df][i]=fr"{image_folder_path}\{data[images_name][i]}"
    return df

# this function inserts images into a list
def images_into_array(data:dict,array:list)->list:
    for i, flat_im in enumerate(data[p.cifar_images]):
        im_channels = []
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024: (j + 1) * 1024].reshape((32, 32))
            )
        # Save the original image
        array.append(np.dstack((im_channels)))
    return array

# this function inserts data into csv
def data_to_csv(data:pd.DataFrame,path):
    if (os.path.exists(path) and os.path.isfile(path)):
        os.remove(path)
        print("file deleted")
    data.to_csv(path)

# this function inserts images into a folder
def images_to_folder(images:list, images_names:list):
    for i in range(0, len(images)):
        img = Image.fromarray(images[i])
        img.save(f"{p.image_folder_path}/{images_names[i]}", format="png")

# this function changes the labels indexes that will fit cifar10
def change_labels(labels:list)->list:
    updated_class=[]
    for index,value in labels.items():
        updated_class.append(p.chosen_classes.index(value)+p.cifar10_num_of_classes)
    return updated_class
    #לסדר את הdict לייבלים והתאמה

# this function deletes images that are not from the chosen classes
def extract_images_of_chosen_classes(df:pd.DataFrame,original_num_of_items,images:list):
    num_of_deleted = 0
    for i in range(original_num_of_items):
        if i not in (df.index):
            images.pop(i - num_of_deleted)
            num_of_deleted += 1
    print()

# this function deletes data that is not from the chosen classes
def extract_data(data:dict,images_array)->pd.DataFrame:
    num_of_items=len(data[p.cifar100_labels])
    # create array of images
    images_into_array(data, images_array)

    cifar100_df:pd.DataFrame=data_to_df(data,p.cifar100_labels,p.cifar_images_name,"cifar100",p.image_folder_path)

    # df of only chosen classes
    cifar100_df_chosen_classes:pd.DataFrame = cifar100_df[cifar100_df.labels.isin(p.chosen_classes)]

    # extract images from the chosen classes
    extract_images_of_chosen_classes(cifar100_df_chosen_classes,num_of_items,images_array)

    # change labels to fit cifar10 labels
    cifar100_df_chosen_classes[p.labels_col_name_df]=change_labels(cifar100_df_chosen_classes[p.labels_col_name_df])


    return cifar100_df_chosen_classes

# this function loads cifar100, extract data from the chosen classes, creates a list of the images and df of the data
def load_cifar100_chosen_classes()->pd.DataFrame:

    # load data from cifar100
    meta_data:dict=load(fr"{p.cifar_100_path}\meta")
    train_data:dict=load(fr"{p.cifar_100_path}\train")
    test_data:dict=load(fr"{p.cifar_100_path}\test")

    # df of data after it was filtered (by classes)

    train_df=extract_data(train_data,train_images)
    test_df=extract_data(test_data,test_images)
    cifar100_df = pd.concat([train_df, test_df])
    return cifar100_df

# this function loads cifar10, creates a list of all images and df of all the data
def load_cifar10()->pd.DataFrame:
    cifar10_data=create_df()

    for batch in p.cifar10_path.glob('data_batch_*'):#param
        # load the data
        batch_data = load(batch)
        # create array of images
        images_into_array(batch_data,cifar10_images)#לבדוק איפה Kלשמור תמונות
        # organize the data
        batch_data:pd.DataFrame=data_to_df(batch_data,p.cifar10_labels,p.cifar_images_name,"cifar10",p.image_folder_path)
        cifar10_data:pd.DataFrame=pd.concat([cifar10_data,batch_data])

    # load test batch
    test_batch=load(fr"{p.cifar10_path}\test_batch")
    images_into_array(test_batch, cifar10_images)
    test_batch: pd.DataFrame = data_to_df(test_batch, p.cifar10_labels, p.cifar_images_name, "cifar10",p.image_folder_path)
    cifar10_data: pd.DataFrame = pd.concat([cifar10_data, test_batch])

    return cifar10_data

# this function separates the data to train, validation and test
def split_data(path):
    data_from_csv = pd.read_csv(path)
    cifar10_df=data_from_csv[data_from_csv[p.dataset_col_name_df]=="cifar10"]
    #train/validation/test - cifar10
    train_cifar10, validation_cifar10, test_cifar10 = np.split(cifar10_df.sample(frac=1, random_state=42),[int(.6*len(cifar10_df)), int(.8*len(cifar10_df))])
    cifar100_df = data_from_csv[data_from_csv[p.dataset_col_name_df] == "cifar100"]
    # train/validation/test - cifar100
    train_cifar100, validation_cifar100, test_cifar100 = np.split(cifar100_df.sample(frac=1, random_state=42),[int(.6 * len(cifar100_df)), int(.8 * len(cifar100_df))])
    return pd.concat([train_cifar10,train_cifar100]),pd.concat([validation_cifar10,validation_cifar100]),pd.concat([test_cifar10,test_cifar100])

# this function inserts the images into ndarray
def data_to_matrix(data:pd.DataFrame)->np.ndarray:
    images=[]
    for row in data.iterrows():
        images.append(load_image(row[1][p.path_col_name_df]))
    img=np.array(images)
    return img

def extract_column(df:pd.DataFrame,col_name):
    return df[col_name]

def main():

    # load cifar10 and cifar100
    cifar10_data:pd.DataFrame=load_cifar10()
    cifar100_data:pd.DataFrame=load_cifar100_chosen_classes()
    cifar100_images=train_images+test_images

    # concat dataframes and images-lists
    all_data:pd.DataFrame=pd.concat([cifar10_data,cifar100_data])
    all_data.reset_index(inplace=True)
    images:np.array=cifar10_images+cifar100_images

    # insert images into a folder
    images_to_folder(images,all_data[p.images_col_name_df])

    # insert the data into csv file
    data_to_csv(all_data,p.csv_path)

    # split to train, test, validation
    train, validation, test=split_data(p.csv_path)

    # create matrix for each part (train, validation and test)
    x_train:np.ndarray=data_to_matrix(train)
    y_train=extract_column(train,p.labels_col_name_df)
    x_validation:np.ndarray=data_to_matrix(validation)
    y_validation=extract_column(validation,p.labels_col_name_df)
    x_test:np.ndarray=data_to_matrix(test)
    y_test=extract_column(test,p.labels_col_name_df)

    # save train, test and validation in npz file
    np.savez(p.binary_file_path, x_train=x_train, y_train=y_train, x_validation=x_validation, y_validation=y_validation, x_test=x_test,
             y_test=y_test)



if __name__=="__main__":
    main()