from pathlib import Path
import pandas as pd

# pathes


cifar10_path = Path(r"C:\Users\IMOE001\Desktop\studied\aplied_material\project\cifar-10-batches-py")
cifar_100_path = Path(r"C:\Users\IMOE001\Desktop\studied\aplied_material\project\cifar-100-python")
image_folder_path = Path(r"C:\Users\IMOE001\Desktop\studied\aplied_material\project\images")
csv_path = Path(r"C:\Users\IMOE001\Desktop\studied\aplied_material\project\cifar.csv")
con_images_f_path = Path(r"C:\Users\IMOE001\Desktop\studied\aplied_material\project\convert_image")
new_images_csv_path = Path(r"C:\Users\IMOE001\Desktop\studied\aplied_material\project\new_images.csv")
new_images_folder_path=Path(r"C:\Users\IMOE001\Desktop\studied\aplied_material\project\new_images")
binary_file_path=Path(r"C:\Users\IMOE001\Desktop\studied\aplied_material\project\data.npz")



chosen_classes = [6, 8, 9, 14, 17]

# for csv1/DataFrame
labels_col_name_df = 'labels'
images_col_name_df = 'image_name'
dataset_col_name_df = 'dataset'
path_col_name_df = 'image path'

# for csv2
label_col_name = 'labels'
image_col_name = 'image_name'
path_col_name = 'image path'
image_col_ind=1

# cifar100
cifar100_labels = b'coarse_labels'

# cifar10
cifar10_labels = b'labels'
cifar10_num_of_classes = 10
cifar10_image_size=[32,32]

# cifar
cifar_images_name = b'filenames'
cifar_images = b'data'

#labels list
labels=['airplane', 'automobile', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
'household furniture','large carnivores',
'large man-made outdoor things',
'people','trees']

#visualization
width = 1/4