import os


data_dir = os.path.join(os.curdir, "Data")
mask_dir, image_dir, save_dir = os.path.join(data_dir, "Mask"), os.path.join(data_dir, "Images"), os.path.join(data_dir, "TFRecord")