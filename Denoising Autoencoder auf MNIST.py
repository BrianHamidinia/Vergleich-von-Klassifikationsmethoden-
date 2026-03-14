import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tensorflow.keras.utils import plot_model
import h5py
from sklearn.model_selection import train_test_split
import os

# ============= 1. Load MNIST & Split =================
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
# --- reshape to (N, 28, 28, 1) and normalize to [0,1]
X_train_full = X_train_full.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train_full = np.expand_dims(X_train_full, -1)
X_test = np.expand_dims(X_test, -1)

# --- Split train_full -> train, val (completely separate) ---
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.1,
    stratify=y_train_full,
    shuffle=True,
)

# ====== 2. Add Gaussian noise (done separately for each split) ======
noise_factor = 0.3

def add_noise(X, noise_factor=0.3, batch_size=64):
    X_noisy = np.empty_like(X)
    for i in range(0, X.shape[0], batch_size):
        X_noisy[i:i+batch_size] = X[i:i+batch_size] + noise_factor * np.random.normal(0, 1, X[i:i+batch_size].shape)
    return np.clip(X_noisy, 0., 1.)

X_train_noisy = add_noise(X_train, noise_factor)
X_val_noisy   = add_noise(X_val, noise_factor)
X_test_noisy  = add_noise(X_test, noise_factor)

# =========== 3. Define Autoencoder ==============
def unet_autoencoder(input_shape=(28,28,1)):
    inputs = Input(shape=input_shape)
    # Encoder
    c1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2,2))(c1) # -> (14,14)
    p1 = Dropout(0.2)(p1)

    c2 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2,2))(c2) # -> (7,7)
    p2 = Dropout(0.2)(p2)

    # Bottleneck
    b = Conv2D(8, (3,3), activation='relu', padding='same')(p2)
    b = BatchNormalization()(b)
    bottleneck = Dropout(0.2)(b)

    # Decoder
    u2 = UpSampling2D((2,2))(bottleneck) # -> (14,14)
    u2 = Concatenate()([u2, c2])
    d2 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(u2)
    d2 = BatchNormalization()(d2)
    d2 = Dropout(0.2)(d2)

    u1 = UpSampling2D((2,2))(d2) # -> (28,28)
    u1 = Concatenate()([u1, c1])
    d1 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(u1)
    d1 = BatchNormalization()(d1)
    outputs = Conv2D(1, (3,3), activation='sigmoid', padding='same')(d1)
    autoencoder = Model(inputs, outputs)
    encoder = Model(inputs, bottleneck)
    return autoencoder, encoder


autoencoder, encoder = unet_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

# =========== 4. Train (Unchanged) ============
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]
history = autoencoder.fit(
    X_train_noisy, X_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(X_val_noisy, X_val),
    callbacks=callbacks,
    verbose=2
)

# =========== 5. Visualization (3-row: org, noisy, denoised) ==============
def plot_3row_images(X_org, X_noisy, X_denoised, y_labels, n_classes=10, n_per_class=1):
    fig, axes = plt.subplots(3, n_classes, figsize=(2*n_classes, 6))
    plt.suptitle("Original (top) | Noisy (middle) | Denoised (bottom) for each class", fontsize=14)
    for digit in range(n_classes):
        idxs = np.where(y_labels.flatten() == digit)[0][:n_per_class]
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        axes[0, digit].imshow(X_org[idx].squeeze(), cmap='gray')
        axes[0, digit].axis('off')
        axes[0, digit].set_title(f"Class {digit}", fontsize=10)
        axes[1, digit].imshow(X_noisy[idx].squeeze(), cmap='gray')
        axes[1, digit].axis('off')
        axes[2, digit].imshow(X_denoised[idx].squeeze(), cmap='gray')
        axes[2, digit].axis('off')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

# Reconstruction on test (batchwise)
def predict_in_batches(model, X, batch_size=256):
    outputs = []
    for i in range(0, X.shape[0], batch_size):
        out = model.predict(X[i:i+batch_size], verbose=0)
        outputs.append(out)
    return np.concatenate(outputs, axis=0)

decoded_test = predict_in_batches(autoencoder, X_test_noisy, batch_size=256)
decoded_test = np.clip(decoded_test, 0., 1.)

plot_3row_images(X_test, X_test_noisy, decoded_test, y_test, n_classes=10, n_per_class=1)

# ========== 6. FEATURE EXTRACTION, SAVE COMPLETELY SEPARATE ==============
def extract_features_batched(encoder, X, batch_size=64):
    features = []
    for i in range(0, len(X), batch_size):
        features.append(encoder.predict(X[i:i+batch_size], verbose=0))
    return np.concatenate(features, axis=0)

# --- Extract features for each split, completely separate ---
latent_train = extract_features_batched(encoder, X_train_noisy, batch_size=64)
latent_val   = extract_features_batched(encoder, X_val_noisy,   batch_size=64)
latent_test  = extract_features_batched(encoder, X_test_noisy,  batch_size=64)

latent_train_flat = latent_train.reshape(latent_train.shape[0], -1)
latent_val_flat   = latent_val.reshape(latent_val.shape[0],   -1)
latent_test_flat  = latent_test.reshape(latent_test.shape[0],  -1)

# ---- SAVE FEATURES, FULLY SEPARATED ----
os.makedirs("features_cae_mnist", exist_ok=True)
with h5py.File('features_cae_mnist/cae_feat_train_mnist.h5', 'w') as hf:
    hf.create_dataset("features", data=latent_train_flat)
    hf.create_dataset("labels", data=y_train)
with h5py.File('features_cae_mnist/cae_feat_val_mnist.h5', 'w') as hf:
    hf.create_dataset("features", data=latent_val_flat)
    hf.create_dataset("labels", data=y_val)
with h5py.File('features_cae_mnist/cae_feat_test_mnist.h5', 'w') as hf:
    hf.create_dataset("features", data=latent_test_flat)
    hf.create_dataset("labels", data=y_test)

print("تمام ویژگی‌های train، val و test روی MNIST به صورت کاملاً جدا ذخیره شدند.")

# ----------- 7. MSE (Reconstruction Loss) per class 
def mean_squared_error_batchwise(X, Y, batch_size=256):
    mse_list = []
    for i in range(0, X.shape[0], batch_size):
        mse = mean_squared_error(X[i:i+batch_size].reshape(len(X[i:i+batch_size]), -1),
                                 Y[i:i+batch_size].reshape(len(Y[i:i+batch_size]), -1))
        mse_list.append(mse)
    return np.mean(mse_list)

# --- Reconstruction Loss (MSE) on the whole test set ---
overall_loss = mean_squared_error_batchwise(X_test, decoded_test)
print(f"Reconstruction Loss (MSE) on the whole test set: {overall_loss:.6f}")

# --- Mean PSNR & SSIM on test set ---
def psnr_ssim_batchwise(X, Y, batch_size=128):
    psnr_list, ssim_list = [], []
    for i in range(0, X.shape[0], batch_size):
        for j in range(X[i:i+batch_size].shape[0]):
            orig = X[i+j]
            recon = Y[i+j]
            psnr_val = psnr(orig, recon, data_range=1.0)
            ssim_val = ssim(orig, recon, data_range=1.0, channel_axis=-1)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
    return np.mean(psnr_list), np.mean(ssim_list)

mean_psnr, mean_ssim = psnr_ssim_batchwise(X_test, decoded_test)
print(f"Mean PSNR on test set: {mean_psnr:.3f}")
print(f"Mean SSIM on test set: {mean_ssim:.3f}")

# --- MSE per class ---
loss_per_class = []
for cls in range(10):
    idxs = np.where(y_test.flatten() == cls)[0]
    loss = mean_squared_error_batchwise(X_test[idxs], decoded_test[idxs])
    loss_per_class.append(loss)
print("\nReconstruction Loss (MSE) per class:")
print("Class | Loss")
print("------|------------")
for digit, loss in enumerate(loss_per_class):
    print(f"  {digit}   | {loss:.6f}")

plt.figure(figsize=(8, 4))
plt.bar(range(10), loss_per_class)
plt.xlabel("Class")
plt.ylabel("Reconstruction Loss (MSE)")
plt.title("Reconstruction Loss (MSE) per Class")
plt.show()

# ----------- 8. (Train vs Val) -----------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Training & Validation Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
