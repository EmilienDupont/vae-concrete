from util import get_processed_mnist
from vae_concrete import VAE

(x_train, y_train), (x_test, y_test) = get_processed_mnist()
model = VAE(latent_disc_dim=10)
model.fit(x_train, num_epochs=1)
model.plot(std_dev=1.)
