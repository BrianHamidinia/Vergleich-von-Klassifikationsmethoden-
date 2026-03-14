import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

# ========== Configuration ==========
config = {
    "num_epochs": 500,
    "batch_size": 128,
    "noise_dim": 100,
    "num_examples_to_generate": 16,
    "g_learning_rate": 0.0003,
    "d_learning_rate": 0.0003,
    "output_dir": os.path.join(os.path.expanduser('~'), 'Desktop', 'dcgan_cifar10_results_v3'),
    "gif_path": os.path.join(os.path.expanduser('~'), 'Desktop', 'dcgan_cifar10_training_v3.gif'),
}
os.makedirs(config["output_dir"], exist_ok=True)

# ========== Load & split CIFAR-10 ==========
(x_train_full, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


x_train_full = (x_train_full - 0.5) * 2.0
x_test = (x_test - 0.5) * 2.0

# Split 10% from train for validation
x_train, x_val = train_test_split(
    x_train_full, test_size=0.1, shuffle=True
)

# ========== Folders ==========
train_dir = os.path.join(config["output_dir"], "train_daten")
test_dir = os.path.join(config["output_dir"], "test_daten")
png_fake_dir = os.path.join(config["output_dir"], "png_fake")
png_real_dir = os.path.join(config["output_dir"], "png_real")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(png_fake_dir, exist_ok=True)
os.makedirs(png_real_dir, exist_ok=True)

# ========== Generator ==========
def build_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(config["noise_dim"],)),
        layers.Dense(4*4*512, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((4, 4, 512)),
        layers.Conv2DTranspose(256, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(3, 5, strides=1, padding='same', use_bias=False, activation='tanh')
    ])
    return model

# ========== Discriminator ==========
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(64, 5, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.4),
        layers.Conv2D(128, 5, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.4),
        layers.Conv2D(256, 5, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# ========== Loss Functions ==========
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output)*0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# ========== DCGAN Model ==========
class DCGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, noise_dim):
        super(DCGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = noise_dim

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.g_loss_fn(fake_output)
            disc_loss = self.d_loss_fn(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return {"d_loss": disc_loss, "g_loss": gen_loss}

# ========== Callback for Loss History ==========
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.g_losses = []
        self.d_losses = []
    def on_epoch_end(self, epoch, logs=None):
        self.g_losses.append(logs['g_loss'])
        self.d_losses.append(logs['d_loss'])

loss_history = LossHistory()

# ========== Callback for Saving Fakes/Reals as PNG and final NPY ==========
class CustomImageSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, generator, train_data, test_data, noise_dim, png_fake_dir, png_real_dir):
        super().__init__()
        self.generator = generator
        self.noise_dim = noise_dim
        self.png_fake_dir = png_fake_dir
        self.png_real_dir = png_real_dir
        self.epoch_save_points = [0] + [e for e in range(49, config['num_epochs'], 50)]  # 1,50,100...
        self.n_images = 16
        self.seed = tf.random.normal([self.n_images, self.noise_dim])
        self.train_data = train_data
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epoch_save_points:
            # Generate fake images for PNG grid
            fake_images = self.generator(self.seed, training=False)
            fake_images = (fake_images + 1.0) / 2.0
            self.save_image_grid(fake_images.numpy(), f"fake_epoch{epoch+1:03d}.png", self.png_fake_dir)

            # Save real train images as PNG grid
            idx = np.random.choice(self.train_data.shape[0], self.n_images, replace=False)
            real_samples = self.train_data[idx]
            real_samples = (real_samples + 1.0) / 2.0
            self.save_image_grid(real_samples, f"real_train_epoch{epoch+1:03d}.png", self.png_real_dir)

            # Save real test images as PNG grid
            idx_test = np.random.choice(self.test_data.shape[0], self.n_images, replace=False)
            real_samples_test = self.test_data[idx_test]
            real_samples_test = (real_samples_test + 1.0) / 2.0
            self.save_image_grid(real_samples_test, f"real_test_epoch{epoch+1:03d}.png", self.png_real_dir)

    def save_image_grid(self, images, filename, out_dir):
        rows = cols = int(np.ceil(np.sqrt(images.shape[0])))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()
        for img, ax in zip(images, axes):
            ax.imshow(img)
            ax.axis('off')
        # Hide any unused subplots
        for ax in axes[images.shape[0]:]:
            ax.axis('off')
        plt.tight_layout()
        filepath = os.path.join(out_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"Saved PNG: {filepath}")

    def on_train_end(self, logs=None):
        # Save final fake features for train and test as numpy arrays
        # Use full batch, no need to keep labels, just images
        print('Extracting final fake features for CNN...')
        # Train fake features
        train_fake = []
        for i in range(0, self.train_data.shape[0], 256):
            batch = min(256, self.train_data.shape[0] - i)
            noise = tf.random.normal([batch, self.noise_dim])
            fake = self.generator(noise, training=False).numpy()
            fake = (fake + 1.0) / 2.0
            train_fake.append(fake)
        train_fake = np.concatenate(train_fake, axis=0)
        np.save(os.path.join(train_dir, "fake_train.npy"), train_fake)
        print(f"Saved final fake train data: {os.path.join(train_dir, 'fake_train.npy')}")
        # Test fake features
        test_fake = []
        for i in range(0, self.test_data.shape[0], 256):
            batch = min(256, self.test_data.shape[0] - i)
            noise = tf.random.normal([batch, self.noise_dim])
            fake = self.generator(noise, training=False).numpy()
            fake = (fake + 1.0) / 2.0
            test_fake.append(fake)
        test_fake = np.concatenate(test_fake, axis=0)
        np.save(os.path.join(test_dir, "fake_test.npy"), test_fake)
        print(f"Saved final fake test data: {os.path.join(test_dir, 'fake_test.npy')}")

# ========== Build and Compile Model ==========
generator = build_generator()
discriminator = build_discriminator()
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=config["g_learning_rate"], beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=config["d_learning_rate"], beta_1=0.5)

dcgan = DCGAN(generator, discriminator, config["noise_dim"])
dcgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    d_loss_fn=discriminator_loss,
    g_loss_fn=generator_loss
)

custom_callback = CustomImageSaveCallback(
    generator, x_train, x_test, config["noise_dim"],
    png_fake_dir, png_real_dir
)

# ========== Training ==========
train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train)
).shuffle(buffer_size=x_train.shape[0]).batch(config["batch_size"])
dcgan.fit(
    train_dataset,
    epochs=config["num_epochs"],
    callbacks=[custom_callback, loss_history],
    verbose=2
)

# ========== Plot Loss ==========
plt.figure(figsize=(10, 5))
plt.plot(loss_history.g_losses, label='Generator Loss')
plt.plot(loss_history.d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DCGAN CIFAR-10 Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(config["output_dir"], "loss_plot.png"))
plt.close()

# ========== t-SNE (batchwise and safe for RAM) ==========
def tsne_real_fake_plot(real, fake, output_path, n_per_class=150):
    idx_real = np.random.choice(real.shape[0], n_per_class, replace=False)
    idx_fake = np.random.choice(fake.shape[0], n_per_class, replace=False)
    real_sample = real[idx_real].reshape(n_per_class, -1)
    fake_sample = fake[idx_fake].reshape(n_per_class, -1)
    X = np.concatenate([real_sample, fake_sample], axis=0)
    y = np.array([0]*n_per_class + [1]*n_per_class)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_emb = tsne.fit_transform(X)
    plt.figure(figsize=(7,7))
    plt.scatter(X_emb[y==0,0], X_emb[y==0,1], s=12, label='REAL', alpha=0.7)
    plt.scatter(X_emb[y==1,0], X_emb[y==1,1], s=12, label='FAKE', alpha=0.7)
    plt.legend()
    plt.title("t-SNE: REAL vs FAKE")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"t-SNE plot saved: {output_path}")

# Example usage after training:
# Load generated fake_train.npy if needed and plot t-SNE
# fake_train_sample = np.load(os.path.join(train_dir, "fake_train.npy"))
# tsne_real_fake_plot(x_train, fake_train_sample, os.path.join(config["output_dir"], "tsne_real_vs_fake.png"), n_per_class=300)
