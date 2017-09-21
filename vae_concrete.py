import numpy as np

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, merge
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras.objectives import binary_crossentropy
from util import (kl_normal, kl_discrete, sampling_normal,
                  sampling_concrete, plot_digit_grid, EPSILON)


class VAE():
    """
    Class to handle building and training VAE models.
    """
    def __init__(self, input_shape=(28, 28, 1), latent_cont_dim=2,
                 latent_disc_dim=0, hidden_dim=128, filters=(64, 64, 64)):
        """
        Setting up everything.

        Parameters
        ----------
        input_shape : Array-like, shape (num_rows, num_cols, num_channels)
            Shape of image.

        latent_cont_dim : int
            Dimension of latent distribution.

        latent_disc_dim : int
            Dimension of discrete latent distribution.

        hidden_dim : int
            Dimension of hidden layer.

        filters : Array-like, shape (num_filters, num_filters, num_filters)
            Number of filters for each convolution in increasing order of
            depth.
        """
        self.opt = None
        self.model = None
        self.input_shape = input_shape
        self.latent_cont_dim = latent_cont_dim
        self.latent_disc_dim = latent_disc_dim
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim
        self.hidden_dim = hidden_dim
        self.filters = filters

    def fit(self, x_train, num_epochs=1, batch_size=100, val_split=.1,
            learning_rate=1e-3, reset_model=True):
        """
        Training model
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if reset_model:
            self._set_model()

        if x_train.shape[0] % batch_size != 0:
            raise(RuntimeError("Training data shape {} is not divisible by batch size {}".format(x_train.shape[0], self.batch_size)))

        # Update parameters
        K.set_value(self.opt.lr, learning_rate)
        self.model.compile(optimizer=self.opt, loss=self._vae_loss)

        self.model.fit(x_train, x_train,
                       epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       validation_split=val_split)

    def _set_model(self):
        """
        Setup model (method should only be called in self.fit())
        """
        print("Setting up model...")
        # Encoder
        inputs = Input(batch_shape=(self.batch_size,) + self.input_shape)

        # Instantiate encoder layers
        Q_0 = Conv2D(self.input_shape[2], (2, 2), padding='same',
                     activation='relu')
        Q_1 = Conv2D(self.filters[0], (2, 2), padding='same', strides=(2, 2),
                     activation='relu')
        Q_2 = Conv2D(self.filters[1], (3, 3), padding='same', strides=(1, 1),
                     activation='relu')
        Q_3 = Conv2D(self.filters[2], (3, 3), padding='same', strides=(1, 1),
                     activation='relu')
        Q_4 = Flatten()
        Q_5 = Dense(self.hidden_dim, activation='relu')
        Q_z_mean = Dense(self.latent_cont_dim)
        Q_z_log_var = Dense(self.latent_cont_dim)

        # Set up encoder
        x = Q_0(inputs)
        x = Q_1(x)
        x = Q_2(x)
        x = Q_3(x)
        flat = Q_4(x)
        hidden = Q_5(flat)

        # Parameters for continous latent distribution
        z_mean = Q_z_mean(hidden)
        z_log_var = Q_z_log_var(hidden)
        # Parameters for concrete latent distribution
        if self.latent_disc_dim:
            Q_c = Dense(self.latent_disc_dim, activation='softmax')
            alpha = Q_c(hidden)

        # Sample from latent distributions
        if self.latent_disc_dim:
            z = Lambda(self._sampling_normal)([z_mean, z_log_var])
            c = Lambda(self._sampling_concrete)(alpha)
            encoding = Concatenate()([z, c])
        else:
            encoding = Lambda(self._sampling_normal)([z_mean, z_log_var])

        # Generator
        # Instantiate generator layers to be able to sample from latent
        # distribution later
        out_shape = (self.input_shape[0] / 2, self.input_shape[1] / 2, self.filters[2])
        G_0 = Dense(self.hidden_dim, activation='relu')
        G_1 = Dense(np.prod(out_shape), activation='relu')
        G_2 = Reshape(out_shape)
        G_3 = Conv2DTranspose(self.filters[2], (3, 3), padding='same',
                              strides=(1, 1), activation='relu')
        G_4 = Conv2DTranspose(self.filters[1], (3, 3), padding='same',
                              strides=(1, 1), activation='relu')
        G_5 = Conv2DTranspose(self.filters[0], (2, 2), padding='valid',
                              strides=(2, 2), activation='relu')
        G_6 = Conv2D(self.input_shape[2], (2, 2), padding='same',
                     strides=(1, 1), activation='sigmoid', name='generated')

        # Apply generator layers
        x = G_0(encoding)
        x = G_1(x)
        x = G_2(x)
        x = G_3(x)
        x = G_4(x)
        x = G_5(x)
        generated = G_6(x)

        self.model = Model(inputs, generated)

        # Set up generator
        inputs_G = Input(batch_shape=(self.batch_size, self.latent_dim))
        x = G_0(inputs_G)
        x = G_1(x)
        x = G_2(x)
        x = G_3(x)
        x = G_4(x)
        x = G_5(x)
        generated_G = G_6(x)
        self.generator = Model(inputs_G, generated_G)

        # Store latent distribution parameters
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        if self.latent_disc_dim:
            self.alpha = alpha

        # Compile models
        self.opt = RMSprop()
        self.model.compile(optimizer=self.opt, loss=self._vae_loss)
        # Loss and optimizer do not matter here as we do not train these models
        self.generator.compile(optimizer=self.opt, loss='mse')

        print("Completed model setup.")

    def generate(self, latent_sample):
        """
        Generating examples from samples from the latent distribution.
        """
        # Model requires batch_size batches, so tile if this is not the case
        if latent_sample.shape[0] != self.batch_size:
            latent_sample = np.tile(latent_sample, self.batch_size).reshape(
                              (self.batch_size, self.latent_dim))
        return self.generator.predict(latent_sample, batch_size=self.batch_size)

    def plot(self, std_dev=2.):
        """
        Method to plot generated digits on a grid.
        """
        return plot_digit_grid(self, std_dev=std_dev)

    def _vae_loss(self, x, x_generated):
        """
        Variational Auto Encoder loss.
        """
        x = K.flatten(x)
        x_generated = K.flatten(x_generated)
        reconstruction_loss = self.input_shape[0] * self.input_shape[1] * \
                                  binary_crossentropy(x, x_generated)
        kl_normal_loss = kl_normal(self.z_mean, self.z_log_var)
        if self.latent_disc_dim:
            kl_disc_loss = kl_discrete(self.alpha)
        else:
            kl_disc_loss = 0
        return reconstruction_loss + kl_normal_loss + kl_disc_loss

    def _sampling_normal(self, args):
        """
        Sampling from a normal distribution.
        """
        z_mean, z_log_var = args
        return sampling_normal(z_mean, z_log_var, (self.batch_size, self.latent_cont_dim))

    def _sampling_concrete(self, args):
        """
        Sampling from a concrete distribution
        """
        return sampling_concrete(args, (self.batch_size, self.latent_disc_dim))
