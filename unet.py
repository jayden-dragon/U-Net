from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, BatchNormalization, \
    Dropout, Activation, UpSampling2D, Concatenate
import matplotlib.pyplot as plt
import numpy as np

class UNET(models.Model):
    def conv(x, n_f, mp_flag=True):
        x = MaxPooling2D((2, 2), padding='same')(x) if mp_flag else x
        x = Conv2D(n_f, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Dropout(0.05)(x)
        x = Conv2D(n_f, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        return x

    def deconv_unet(x, e, n_f): # Two inputs, x and e!
        x = UpSampling2D((2, 2))(x)
        x = Concatenate(axis=3)([x, e]) # most important!
        x = Conv2D(n_f, (3, 3), padding='same')(x)
        x = BatchNormalization()(x) # learning booster
        x = Activation('tanh')(x)
        x = Conv2D(n_f, (3, 3), padding='same')(x)
        x = BatchNormalization()(x) # learning booster
        x = Activation('tanh')(x)
        return x

    def __init__(self, org_shape):

        # Input
        original = Input(shape=org_shape)

        # Encoding
        c1 = UNET.conv(original, 56, mp_flag=False)
        c2 = UNET.conv(c1, 32)

        # Encoded vector
        encoded = UNET.conv(c2, 16)

        # Decoding
        x = UNET.deconv_unet(encoded, c2, 32)
        y = UNET.deconv_unet(x, c1, 56)

        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(y)

        super().__init__(original, decoded)
        self.compile(optimizer='adadelta', loss='mse')

class DATA():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        self.x_train_in = x_train
        self.x_test_in = x_test
        self.x_train_out = x_train
        self.x_test_out = x_test

        img_rows, img_cols, n_ch = self.x_train_in.shape[1:]
        self.input_shape = (img_rows, img_cols, n_ch)

def main(in_ch=1, epochs=10, batch_size=512, fig=True):
    data = DATA()
    unet = UNET(data.input_shape)
    unet.summary()

    history = unet.fit(data.x_train_in, data.x_train_out,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_data=(data.x_test_in, data.x_test_out))

    def plot_loss(h, title="loss"):
        plt.plot(h.history['loss'])
        plt.plot(h.history['val_loss'])
        plt.title(title)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc=0)

    def show_images(data, unet):
        x_test_in = data.x_test_in
        x_test_out = data.x_test_out
        decoded_imgs = unet.predict(x_test_in)

        n = 10
        plt.figure(figsize=(20, 6))
        for i in range(n):
            ax = plt.subplot(3, n, i+1)
            plt.imshow(x_test_in[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, n, i+n)
            plt.imshow(decoded_imgs[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, n, i+1+n*2)
            plt.imshow(x_test_out[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    if fig:
        plot_loss(history)
        plt.savefig('unet.loss.png')
        plt.clf()
        show_images(data, unet)
        plt.savefig('unet.pred.png')

if __name__ == '__main__':
    import argparse
    from distutils import util

    parser = argparse.ArgumentParser(description='UNET for Cifar-10')
    parser.add_argument('--epochs', type=int, default=200,
                        help='training epochs (default: 200')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128')
    parser.add_argument('--fig', type=lambda x: bool(util.strtobool(x)),
                        default=True, help='flag to show figures (default: True)')
    args = parser.parse_args()
    print("Aargs:", args)

    main(args.epochs, args.batch_size, args.fig)