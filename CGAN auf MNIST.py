
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt

# ==== 1. Load and prepare MNIST data ====
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  # Changed to also load test data

x_train = x_train.astype('float32') / 255.0
x_train = x_train * 2. - 1.   # scale to [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)

x_test = x_test.astype('float32') / 255.0
x_test = x_test * 2. - 1.
x_test = np.expand_dims(x_test, axis=-1)

num_classes = 10
latent_dim = 100
img_shape = (28, 28, 1)
BATCH_SIZE = 64
EPOCHS = 100
OUTPUT_DIR = "./cgan_mnist_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== 2. Define the Generator ====
def build_generator(latent_dim, num_classes):
    label = layers.Input(shape=(1,))
    label_embedding = layers.Embedding(num_classes, 50)(label)
    n_nodes = 7 * 7
    label_dense = layers.Dense(n_nodes)(label_embedding)
    label_reshaped = layers.Reshape((7, 7, 1))(label_dense)

    latent = layers.Input(shape=(latent_dim,))
    gen_dense = layers.Dense(128 * 7 * 7)(latent)
    gen_dense = layers.LeakyReLU(alpha=0.2)(gen_dense)
    gen_reshaped = layers.Reshape((7, 7, 128))(gen_dense)

    merged = layers.Concatenate()([gen_reshaped, label_reshaped])
    gen = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same',
                                 activation=layers.LeakyReLU(alpha=0.2))(merged)
    gen = layers.BatchNormalization()(gen)
    gen = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same',
                                 activation=layers.LeakyReLU(alpha=0.2))(gen)
    gen = layers.BatchNormalization()(gen)
    out_layer = layers.Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
    model = keras.Model([latent, label], out_layer, name="generator")
    return model

generator = build_generator(latent_dim, num_classes)

# ==== 3. Define the Discriminator ====
def build_discriminator(img_shape, num_classes):
    label = layers.Input(shape=(1,))
    label_embedding = layers.Embedding(num_classes, 50)(label)
    n_nodes = img_shape[0] * img_shape[1]
    label_dense = layers.Dense(n_nodes)(label_embedding)
    label_reshaped = layers.Reshape((img_shape[0], img_shape[1], 1))(label_dense)

    img = layers.Input(shape=img_shape)
    merged = layers.Concatenate()([img, label_reshaped])
    fe = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same',
                       activation=layers.LeakyReLU(alpha=0.2))(merged)
    fe = layers.Dropout(0.4)(fe)
    fe = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same',
                       activation=layers.LeakyReLU(alpha=0.2))(fe)
    fe = layers.Dropout(0.4)(fe)
    fe = layers.Flatten()(fe)
    out_layer = layers.Dense(1, activation='sigmoid')(fe)
    model = keras.Model([img, label], out_layer, name="discriminator")
    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

discriminator = build_discriminator(img_shape, num_classes)

# ==== 4. Define the cGAN ====
def build_gan(generator, discriminator):
    discriminator.trainable = False
    noise, label = generator.input
    img = generator.output
    gan_output = discriminator([img, label])
    model = keras.Model([noise, label], gan_output)
    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

gan = build_gan(generator, discriminator)

# ==== 5. Utility Functions ====
def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    idx = np.random.randint(0, images.shape[0], n_samples)
    X, labels = images[idx], labels[idx]
    y = np.ones((n_samples, 1))
    return [X, labels], y

def generate_noise(latent_dim, n_samples, n_classes=10):
    z_input = np.random.randn(n_samples, latent_dim)
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
    z_input, labels_input = generate_noise(latent_dim, n_samples)
    images = generator.predict([z_input, labels_input], verbose=0)
    y = np.zeros((n_samples, 1))
    return [images, labels_input], y

def save_image_grid(images, n, epoch, output_dir):
    images = (images + 1) / 2.0  # scale from [-1,1] to [0,1]
    plt.figure(figsize=(n, n))
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.axis('off')
        plt.imshow(images[i, :, :, 0], cmap='gray_r')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"generated_images_epoch_{epoch:03d}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def train_cgan(generator, discriminator, gan, dataset, latent_dim, n_epochs=100, n_batch=128):
    X_train, y_train = dataset
    steps_per_epoch = int(X_train.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    g_losses = []
    d_losses = []
    for epoch in range(n_epochs):
        for step in range(steps_per_epoch):
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = discriminator.train_on_batch([X_real, labels_real], y_real)
            [X_fake, labels_fake], y_fake = generate_fake_samples(generator, latent_dim, half_batch)
            d_loss2, _ = discriminator.train_on_batch([X_fake, labels_fake], y_fake)
            z_input, labels_input = generate_noise(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan.train_on_batch([z_input, labels_input], y_gan)
            d_losses.append(0.5 * (d_loss1 + d_loss2))
            g_losses.append(g_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, d1={d_loss1:.3f}, d2={d_loss2:.3f}, g={g_loss:.3f}")
        if (epoch+1) % 10 == 0:
            [X_fake, _], _ = generate_fake_samples(generator, latent_dim, 25)
            save_image_grid(X_fake, 5, epoch+1, OUTPUT_DIR)

    generator.save(os.path.join(OUTPUT_DIR, 'cgan_generator.h5'))
    plt.figure(figsize=(8, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('CGAN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cgan_loss_curve.png"))
    plt.close()


train_cgan(generator, discriminator, gan, [x_train, y_train], latent_dim, n_epochs=EPOCHS, n_batch=BATCH_SIZE)


def create_fake_dataset_for_labels(generator, latent_dim, labels, prefix, output_dir, batch_size=128):
    n_samples = len(labels)
    labels_flat = labels.flatten() if labels.ndim > 1 else labels
    img_shape = (28, 28, 1)
    fake_images_file = os.path.join(output_dir, f'fake_mnist_images_{prefix}.npy')
    labels_file = os.path.join(output_dir, f'fake_mnist_labels_{prefix}.npy')
   
    fake_images_memmap = np.lib.format.open_memmap(
        fake_images_file, mode='w+', dtype=np.float32, shape=(n_samples, *img_shape)
    )
    for i in range(0, n_samples, batch_size):
        end_i = min(i + batch_size, n_samples)
        batch_labels = labels_flat[i:end_i]
        z = np.random.randn(end_i - i, latent_dim)
        batch_fake = generator.predict([z, batch_labels], verbose=0)
        fake_images_memmap[i:end_i] = batch_fake  

    np.save(labels_file, labels_flat)
    print(f"Saved: {fake_images_file}, {labels_file}")

# Load the trained generator
generator = keras.models.load_model(os.path.join(OUTPUT_DIR, 'cgan_generator.h5'))

# Generate and shuffle fake dataset for train data (same size as x_train)
create_fake_dataset_for_labels(generator, latent_dim, y_train, prefix='train', output_dir=OUTPUT_DIR)

# Generate and shuffle fake dataset for test data (same size as x_test)
create_fake_dataset_for_labels(generator, latent_dim, y_test, prefix='test', output_dir=OUTPUT_DIR)


