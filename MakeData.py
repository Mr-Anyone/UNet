import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.asdftrain import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example


def init(paths):
    for path in paths:
        try:
            os.mkdir(path)
        except Exception as e:
            pass


def find_image_path(filename, image_dir):
    filename = filename.split(".")[0]
    for name in os.listdir(image_dir):
        if filename in name:
            return os.path.join(image_dir, name)


def load_img(path, color_mode="rgb"):
    img = keras.preprocessing.image.load_img(path, color_mode=color_mode)
    return keras.preprocessing.image.img_to_array(img)


def save(mask, img, save_dir, count=0):
    mask = tf.io.encode_jpeg(mask)
    img = tf.io.encode_jpeg(img)

    Image_Buffer = Example(
        features=Features(feature={
            "Image": Feature(bytes_list=BytesList(value=[img.numpy()])),
            "Mask": Feature(bytes_list=BytesList(value=[mask.numpy()]))
        })
    )

    with tf.io.TFRecordWriter(os.path.join(save_dir, f"Data-{count}.tfrecord")) as f:
        f.write(Image_Buffer.SerializeToString())


def make_unet_tfrecord(mask_dir, image_dir, save_dir, img_size=(256, 256)):
    count = 0

    init([mask_dir, image_dir, save_dir])

    for filename in os.listdir(mask_dir):
        try:
            path = os.path.join(mask_dir, filename)
            img_path = find_image_path(filename, image_dir)

            mask = load_img(path, color_mode='grayscale')
            img = load_img(img_path)

            img = tf.image.resize(img, img_size)
            img = tf.cast(img, tf.uint8)

            mask = tf.image.resize(mask, img_size)
            mask = tf.cast(mask, tf.uint8)

            save(mask, img, save_dir, count=count)

            if count % 100 == 0:
                print(f"Processed {count} Images")
            count += 1
        except Exception as e:
            pass


if __name__ == "__main__":
    from Config import mask_dir, tf_records_path, image_dir
    print("Starting To Make TF-Record")
    make_tf_record(mask_dir, image_dir, tf_records_path)