import tensorflow as tf
from tensorflow import keras


class UNet(keras.Model):
    def __init__(self, input_shape, depth=5, base_filter=16, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        filter_counts = [base_filter]
        for i in range(depth - 1):
            filter_counts.append(filter_counts[-1] * 2)

        self.u_left_lays = []
        for i in range(depth - 1):  # Creating the left side of the U Shape
            for x in range(3):
                if i == 0 and x == 0:
                    self.u_left_lays.append(
                        keras.layers.Conv2D(filter_counts[i], (3, 3), padding='same', activation='relu',
                                            input_shape=input_shape))
                else:
                    self.u_left_lays.append(
                        keras.layers.Conv2D(filter_counts[i], (3, 3), padding='same', activation='relu'))
            self.u_left_lays.append(keras.layers.MaxPool2D((2, 2), strides=2))

        self.mid_lays = []
        for i in range(3):
            self.mid_lays.append(keras.layers.Conv2D(filter_counts[-1], (3, 3), padding='same', activation='relu'))

        self.u_right_lays = []
        for i in range(depth - 1):
            self.u_right_lays.append(keras.layers.UpSampling2D())
            for x in range(3):
                self.u_right_lays.append(
                    keras.layers.Conv2D(filter_counts[-(i + 2)], (3, 3), padding='same', activation='relu'))

        self.final_layer = keras.layers.Conv2D(1, (1, 1,), padding='same', activation='sigmoid')

    def call(self, Z):
        Z_concats = []

        k = 2  # Concat in layer
        for i in range(len(self.u_left_lays)):
            layer = self.u_left_lays[i]
            Z = layer(Z)

            if i == k:
                Z_concats.append(Z)
                k += 4

        for layer in self.mid_lays:
            Z = layer(Z)

        k = 1
        concat_count = -1
        for i in range(len(self.u_right_lays)):
            layer = self.u_right_lays[i]

            if i == k:
                _, width, height, _ = Z.shape

                Z = tf.concat([Z, Z_concats[concat_count][:, :width, :height]], axis=3)  # Crop Image
                concat_count -= 1
                k += 4

            Z = layer(Z)

        Z = self.final_layer(Z)
        return Z