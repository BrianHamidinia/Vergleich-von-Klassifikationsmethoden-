import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split

# --------- 1. GPU MEMORY GROWTH ---------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except Exception as e:
        print("Could not set GPU memory growth:", e)

# --------- 2. LOAD DATA & SPLIT VALIDATION (completely separated) ---------
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Reshape for Conv2D: (samples, 28, 28, 1) & normalize each split independently
X_train_full = X_train_full.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train_full = np.expand_dims(X_train_full, -1)
X_test = np.expand_dims(X_test, -1)

# Train/Validation split (validation only from train_full, test untouched)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.1,
    stratify=y_train_full,
    random_state=42
)

print("Train set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

# --------- 3. MODEL DEFINITION (CAE, bottleneck=7x7x8=392) ---------
inputs = Input(shape=(28, 28, 1))
x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x1 = BatchNormalization()(x1)
x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
x1 = BatchNormalization()(x1)
p1 = MaxPooling2D((2, 2))(x1)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
x2 = BatchNormalization()(x2)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
x2 = BatchNormalization()(x2)
p2 = MaxPooling2D((2, 2))(x2)
x3 = Conv2D(8, (3, 3), activation='relu', padding='same')(p2)
x3 = BatchNormalization()(x3)
x3 = Dropout(0.3)(x3)
bottleneck = x3
u2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x3)
u2 = BatchNormalization()(u2)
u2 = Concatenate()([u2, x2])
u2 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
u2 = BatchNormalization()(u2)
u2 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
u2 = BatchNormalization()(u2)
u1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(u2)
u1 = BatchNormalization()(u1)
u1 = Concatenate()([u1, x1])
u1 = Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
u1 = BatchNormalization()(u1)
u1 = Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
u1 = BatchNormalization()(u1)
outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(u1)
autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
autoencoder.summary()

# Encoder model
encoder = Model(inputs, bottleneck)

# --------- 4. TRAIN ---------
history = autoencoder.fit(
    X_train, X_train,
    epochs=70,
    batch_size=128,
    shuffle=True,
    validation_data=(X_val, X_val),
    verbose=2
)

# --------- 5. FEATURE EXTRACTION (completely separated for each split) ---------
def extract_features_batched(encoder, X, batch_size=256):
    features = []
    for i in range(0, len(X), batch_size):
        features.append(encoder.predict(X[i:i+batch_size], verbose=0))
    return np.concatenate(features, axis=0)

latent_features_train = extract_features_batched(encoder, X_train, batch_size=256)
latent_features_val   = extract_features_batched(encoder, X_val, batch_size=256)
latent_features_test  = extract_features_batched(encoder, X_test, batch_size=256)

# Reshape bottleneck to (n_samples, -1)
latent_features_train_flat = latent_features_train.reshape(latent_features_train.shape[0], -1)
latent_features_val_flat   = latent_features_val.reshape(latent_features_val.shape[0], -1)
latent_features_test_flat  = latent_features_test.reshape(latent_features_test.shape[0], -1)

# Save each set (features+labels) in SEPARATE h5 files
with h5py.File('cae_feat_train_mnist.h5', 'w') as hf:
    hf.create_dataset("features", data=latent_features_train_flat)
    hf.create_dataset("labels", data=y_train)
with h5py.File('cae_feat_val_mnist.h5', 'w') as hf:
    hf.create_dataset("features", data=latent_features_val_flat)
    hf.create_dataset("labels", data=y_val)
with h5py.File('cae_feat_test_mnist.h5', 'w') as hf:
    hf.create_dataset("features", data=latent_features_test_flat)
    hf.create_dataset("labels", data=y_test)
print("Latent features for train, validation, and test saved separately as cae_feat_train_mnist.h5, cae_feat_val_mnist.h5, cae_feat_test_mnist.h5.")

# --------- 6. RECONSTRUCTION (on test set) ---------
def predict_batches(model, X, batch_size=256):
    result = []
    for i in range(0, len(X), batch_size):
        result.append(model.predict(X[i:i+batch_size], verbose=0))
    return np.concatenate(result, axis=0)

decoded_imgs_test = predict_batches(autoencoder, X_test, batch_size=256)

# --------- 7. RECONSTRUCTION ERROR METRICS (test set only) ---------
X_test_flat = X_test.reshape(-1, 28*28*1)
decoded_imgs_flat = decoded_imgs_test.reshape(-1, 28*28*1)
overall_mse = mean_squared_error(X_test_flat, decoded_imgs_flat)
overall_mae = mean_absolute_error(X_test_flat, decoded_imgs_flat)
print(f"\nReconstruction Loss (MSE) on the whole test set: {overall_mse:.6f}")
print(f"Reconstruction Loss (MAE) on the whole test set: {overall_mae:.6f}")

# MSE per class ONLY (test set)
loss_per_class_mse = []
for digit in range(10):
    idxs = np.where(y_test.flatten() == digit)[0]
    mse = mean_squared_error(X_test_flat[idxs], decoded_imgs_flat[idxs])
    loss_per_class_mse.append(mse)

print("\nReconstruction Loss (MSE) per class:")
print("Class | MSE")
print("------|-----------")
for digit in range(10):
    print(f"  {digit}   | {loss_per_class_mse[digit]:.6f}")

plt.figure(figsize=(8, 4))
plt.bar(range(10), loss_per_class_mse)
plt.xlabel("Class")
plt.ylabel("MSE")
plt.title("Reconstruction Loss (MSE) per Class (Test)")
plt.show()

# --------- 8. Mean PSNR & SSIM (test set only) ---------
psnr_list, ssim_list = [], []
for i in range(0, len(X_test)):
    orig = X_test[i]
    recon = decoded_imgs_test[i]
    psnr_val = psnr(orig, recon, data_range=1.0)
    ssim_val = ssim(orig, recon, data_range=1.0, channel_axis=2)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)
print(f"\nMean PSNR on test set: {np.mean(psnr_list):.3f}")
print(f"Mean SSIM on test set: {np.mean(ssim_list):.3f}")

# --------- 9. Visualize Original vs. Reconstruction for each class (test set) ---------
n_per_class = 2
fig, axes = plt.subplots(2, 10, figsize=(20, 5))
plt.suptitle("Deep CAE (Skip Connections): Original (top) vs. Reconstruction (bottom) per Class (Test)", fontsize=16)
for digit in range(10):
    idxs = np.where(y_test.flatten() == digit)[0][:n_per_class]
    for i, idx in enumerate(idxs):
        axes[0, digit].imshow(X_test[idx].squeeze(), cmap='gray')
        axes[0, digit].axis('off')
        if i == 0:
            axes[0, digit].set_title(f"Class {digit}", fontsize=12)
        axes[1, digit].imshow(decoded_imgs_test[idx].squeeze(), cmap='gray')
        axes[1, digit].axis('off')
plt.show()

# --------- 10. PLOT LOSS CURVE (Train & Validation) ---------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Training & Validation Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
