import csv
from PIL import Image
import numpy as np
import os
from array import *
from pathlib import Path
import pandas as pd
from csv import writer
import params as p
import matplotlib.pyplot as plt


# this function loads an image
def load_image(path)->Image:
    img = plt.imread(path) # Image.open(path)
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
    img = flat_img.reshape(3, p.cifar10_image_size[0], p.cifar10_image_size[1])
    img = img.transpose(1, 2, 0)
    img = Image.fromarray(img)
    return img


# resize an image
def image_to_cifar10_format(image:Image)->Image:
    width, height = image.size
    if width!=height:
        raise Exception("Image is not on size")
    image = image.resize((p.cifar10_image_size[0], p.cifar10_image_size[1]), Image.ANTIALIAS)
    return image

# this function receives details of an image and creates df
def df_for_one_image(label,image_name,path)->pd.DataFrame:
    return pd.DataFrame({p.label_col_name:[label], p.image_col_name: [image_name], p.path_col_name: [path]} )

# this function receives details of an image and add them into a csv
def image_datails_to_csv(label,image_name,image_path,csv_path):
    # check if image name already exists in csv
    if len(find_in_csv(csv_path,p.images_col_name_df,image_name))!=0:
        raise Exception("Image name already exists")
    list = [label,image_name,image_path]
    with open(p.new_images_csv_path, 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list)
        f.close()

# this function receives an image and the image name and insert the image to a folder
def image_to_folder(image:Image,image_name):
    image.save(f'{p.new_images_folder_path}/{image_name}.jpg')

# this function receives col name and value and returns the first row that the value in the col equals to the value that was sent
# אם רוצים לשנות שמקב שם של עמודה ומחפש מה האינדקס ומחפש לפיו
def find_in_csv(path,by_col:int,value)->list:
    csv_file = csv.reader(open(path, "r"))
    for row in csv_file:
        if row[by_col]==value:
            return row
    return []
def extract_from_folder(path)->np.ndarray:
    image=plt.imread(path)
    return image




def main():
    extract_from_folder(r"C:\Users\r0583\Downloads\images.jfif")
    print()



if __name__ == "__main__":

    main()








