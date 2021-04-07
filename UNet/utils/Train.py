from ..lib.Unet import UNet
import tensorflow as tf
import os


def _parse_single_data(X):
    feature_description = {
        "Image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "Mask": tf.io.FixedLenFeature([], tf.string, default_value="")
    }

    data = tf.io.parse_single_example(X, feature_description)
    img = tf.io.decode_jpeg(data["Image"], channels=3)
    mask = tf.io.decode_jpeg(data["Mask"], channels=1)

    return tf.cast(img, tf.float32) / 255., tf.cast(mask, tf.float32) / 255.


def make_dataset(tf_records_path, batch_size=64, validation_split=0.1):
    count = int(len(tf_records_path) * validation_split)
    tf_records_path = [os.path.join(tf_records_path, filename) for filename in os.listdir(tf_records_path)]

    train_data = tf.data.TFRecordDataset(tf_records_path[count:])
    train_data = train_data.map(_parse_single_data)
    train_data = train_data.batch(batch_size)

    valid_data = tf.data.TFRecordDataset(tf_records_path[:count])
    valid_data = valid_data.map(_parse_single_data)
    valid_data = valid_data.batch(batch_size)
    return train_data, valid_data


def train_unet(tf_records_path, epochs=10, batch_size=64, validation_split=0.1, callbacks=None, loss='binary_crossentropy', optimizer='adam', metrics=None):
    if callbacks is None:
        callbacks = []

    if metrics is None:
        metrics = []

    train_data, valid_data = make_dataset(tf_records_path, batch_size, validation_split=validation_split)
    model = UNet(input_shape=[256, 256, 3])
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(train_data, epochs=epochs, callbacks=callbacks, validation_data=valid_data)
    return model


if __name__ == "__main__":
    from .Config import tf_records_path, callbacks, batch_size, epochs, validation_split
    train_unet(tf_records_path, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_split=validation_split, metrics=["accuracy"])