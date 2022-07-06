import csv
from PIL import Image
import numpy as np
import os
from array import *
from pathlib import Path
import pandas as pd
from csv import writer

import matplotlib.pyplot as plt

images_folder_path = Path(r'C:\Users\r0583\Documents\Bootcamp\project\images')
con_images_f_path = Path(r'C:\Users\r0583\Documents\Bootcamp\project\convert_image')
new_images_csv_path = Path(r'C:\Users\r0583\Documents\Bootcamp\project\new_images.csv')
new_images_folder_path=Path(r'C:\Users\r0583\Documents\Bootcamp\project\new_images')
cifar10_image_size=[32,32]

# this function loads an image
def load_image(path)->Image:
    img = Image.open(path)
    return img


# bring an image to a vector
def flatten_image(img:Image)->np.ndarray:
    img = (np.array(img))
    r = img[:, :, 0].flatten()
    g = img[:, :, 1].flatten()
    b = img[:, :, 2].flatten()
    label = [1]
    out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
    return out[1:]


# bring a vector to an image
def flat_image_to_PIL_Image(flat_img:np.ndarray)->Image:
    img = flat_img.reshape(3, cifar10_image_size[0], cifar10_image_size[1])
    img = img.transpose(1, 2, 0)
    img = Image.fromarray(img)
    return img


# resize an image+בדיקה שהתמונה ריבועית להוסיף
def image_to_cifar10_format(image:Image)->Image:
    width, height = image.size
    if width!=height:
        raise Exception("Image is not ")
    image = image.resize((cifar10_image_size[0], cifar10_image_size[1]), Image.ANTIALIAS)
    return image

# this function receives details of an image and creates df
def df_for_one_image(label,image_name,path)->pd.DataFrame:
    return pd.DataFrame({label_col_name:[label], image_col_name: [image_name], path_col_name: [path]} )

# this function receives details of an image and add them into a csv
def image_datails_to_csv(label,image_name,image_path,csv_path):
    # check if image name already exists in csv
    if len(extract_from_csv(csv_path,1,image_name))!=0:
        raise Exception("Image name already exists")
    list = [label,image_name,image_path]
    with open(new_images_csv_path, 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list)
        f.close()

# this function receives an image and the image name and insert the image to a folder
def image_to_folder(image:Image,image_name):
    image.save(f'{new_images_folder_path}/{image_name}.png')

# this function receives col name and value and returns the first row that the value in the col equals to the value that was sent
# אם רוצים לשנות שמקב שם של עמודה ומחפש מה האינדקס ומחפש לפיו
def extract_from_csv(path,by_col:int,value)->list:
    csv_file = csv.reader(open(path, "r"))
    for row in csv_file:
        if row[by_col]==value:
            return row
    return []



def main():
#    images_to_cifar10_format(images_folder_path, con_images_f_path)
    # extract_from_csv(r"C:\Users\r0583\Documents\Bootcamp\project\cifar.csv",2,"b'camion_s_000148.png'")
    # path=r"C:\Users\r0583\Downloads\images.jfif"
    # img=load_image(path)
    # img=image_to_cifar10_format(img)
    image_datails_to_csv('6','flower6','C/fd',new_images_csv_path)
    image_datails_to_csv('6','flower5','C/fd', new_images_csv_path)

# image_to_folder(img,"nice flower")
#     flat_image:np.ndarray=flatten_image(img)
#     img=flat_image_to_PIL_Image(flat_image)


if __name__ == "__main__":
    label_col_name = 'labels'
    image_col_name = 'image_name'
    path_col_name='path'
    main()



# load all images from source folder , convert to cifar10 format and save in dest folder
# def images_to_cifar10_format(source_path, dest_path):
#     for dirname, dirnames, filenames in os.walk(source_path):
#         for filename in filenames:
#             if filename.endswith('.JPG') or filename.endswith('.jpg') or filename.endswith('.pmg'):
#                 img = Image.open(os.path.join(dirname, filename))
#                 img = image_to_cifar10_format(img)
#                 img.save(dest_path + "/" + filename)