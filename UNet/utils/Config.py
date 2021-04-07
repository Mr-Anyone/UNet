import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

data_dir = os.path.join(os.curdir, "Data")
mask_dir, image_dir, tf_records_path = os.path.join(data_dir, "Mask"), os.path.join(data_dir, "Images"), os.path.join(data_dir, "TFRecord")
model_dir = os.path.join(os.curdir, "Models")

batch_size = 64
callbacks = [
    EarlyStopping(patience=10),
    ModelCheckpoint(os.path.join(model_dir, "Best-UNET.h5"), save_best_only=True)
]
validation_split = 0.1
epochs = 1

img_size = (256, 256)