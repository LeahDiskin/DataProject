import pandas as pd
import os
from pathlib import Path

from  Data.labels import cifar10,cifar100,data_to_csv


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