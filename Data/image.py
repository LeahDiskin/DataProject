from PIL import Image
import numpy as np
import os
from array import *
from pathlib import Path

images_folder_path = r'C:\Users\r0583\Documents\Bootcamp\project\new_images\new_images'
con_images_f_path = Path(r'C:\Users\r0583\Documents\Bootcamp\project\new_images\convert_image')
#

# load an image
def load_image(path):
    img = Image.open(path)
    return img


# bring an image to a vector
def flatten_image(img):
    img = (np.array(img))
    r = img[:, :, 0].flatten()
    g = img[:, :, 1].flatten()
    b = img[:, :, 2].flatten()
    label = [1]
    out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
    print()


# bring a vector to an image
def flat_image_to_metrix(flat_img):
    flat_img = flat_img.reshape(3, 32, 32)
    flat_img = flat_img.transpose(1, 2, 0)
    return flat_img


# resize an image to image(32*32)
def image_to_cifar10_format(image):
    image = image.resize((32, 32), Image.ANTIALIAS)


# load all images from source folder , convert to cifar10 format and save in dest folder
def images_to_cifar10_format(source_path, dest_path):
    for dirname, dirnames, filenames in os.walk(source_path):
        for filename in filenames:
            if filename.endswith('.JPG') or filename.endswith('.jpg') or filename.endswith('.pmg'):
                img = Image.open(os.path.join(dirname, filename))
                img = image_to_cifar10_format(img)
                img.save(dest_path + "/" + filename)


# load all images from source folder , convert to cifar10 format and save in dest folder, the function was token from github
def images_to_cifar10_format2(source_path, dest_path):
    data = array('B')
    for dirname, dirnames, filenames in os.walk(r'C:\Users\r0583\Documents\Bootcamp\project\ima'):
        for filename in filenames:
            if filename.endswith('.JPG') or filename.endswith('.jpg') or filename.endswith('.pmg'):
                im = Image.open(os.path.join(dirname, filename))
                pix = im.load()
                data = []
                for color in range(0, 3):
                    for x in range(0, 32):
                        for y in range(0, 32):
                            data.append(pix[x, y][color])
                data = np.array(data)
                data = flat_image_to_metrix(data)
                data.save(dest_path + "/" + filename)


def main():
    images_to_cifar10_format(images_folder_path, con_images_f_path)


if __name__ == "__main__":
    main()