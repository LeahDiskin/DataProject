# load all images from source folder , convert to cifar10 format and save in dest folder
# def images_to_cifar10_format(source_path, dest_path):
#     for dirname, dirnames, filenames in os.walk(source_path):
#         for filename in filenames:
#             if filename.endswith('.JPG') or filename.endswith('.jpg') or filename.endswith('.pmg'):
#                 img = Image.open(os.path.join(dirname, filename))
#                 img = image_to_cifar10_format(img)
#                 img.save(dest_path + "/" + filename)