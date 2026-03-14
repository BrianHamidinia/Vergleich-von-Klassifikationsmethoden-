import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import time
from tensorflow.keras.optimizers import SGD

# 1. Load and preprocess data
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# MNIST: (samples, 28, 28), need to expand dims to (samples, 28, 28, 1) if needed for CNN; for MLP just flatten
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.1,
    stratify=y_train_full,
    random_state=42
)

# Normalize and flatten
X_train = X_train.astype('float32') / 255.0
X_val   = X_val.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

# For MLP: flatten images
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat   = X_val.reshape(X_val.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

y_train_cat = to_categorical(y_train, 10)
y_val_cat   = to_categorical(y_val, 10)
y_test_cat  = to_categorical(y_test, 10)

mnist_labels = [
    '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9'
]

# 2. Data Augmentation (optional and very light for MNIST)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    zoom_range=0.01548,      # same as before
    rotation_range=1,
    width_shift_range=0.04,
    height_shift_range=0.04,
    horizontal_flip=False    # MNIST: horizontal flip not logical
)
datagen.fit(X_train.reshape(-1, 28, 28, 1))  # ImageDataGenerator expects 4D shape

def flatten_generator(generator, X, y, batch_size):
    # For MNIST, need to reshape X to (samples, 28, 28, 1) for augmentation
    X_img = X.reshape(-1, 28, 28, 1)
    gen = generator.flow(X_img, y, batch_size=batch_size, shuffle=True)
    while True:
        X_batch, y_batch = next(gen)
        X_batch_flat = X_batch.reshape(len(X_batch), -1)
        yield X_batch_flat, y_batch

# 3. Define MLP architecture (same as before, but input_shape=(784,))
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(784,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

# 4. Compile with SGD
optimizer = SGD(
    learning_rate=0.0489,
    momentum=0.9217,
    nesterov=True
)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 5. Training
EPOCHS = 120
BATCH_SIZE = 128
train_times = []
history_acc, history_val_acc = [], []
history_loss, history_val_loss = [], []
steps_per_epoch = int(np.ceil(X_train.shape[0] / BATCH_SIZE))
augmented_train_gen = flatten_generator(datagen, X_train, y_train_cat, BATCH_SIZE)

for epoch in range(EPOCHS):
    start_time = time.time()
    hist = model.fit(
        augmented_train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val_flat, y_val_cat),
        epochs=1,
        verbose=2
    )
    end_time = time.time()
    train_times.append(end_time - start_time)
    history_acc.extend(hist.history['accuracy'])
    history_val_acc.extend(hist.history['val_accuracy'])
    history_loss.extend(hist.history['loss'])
    history_val_loss.extend(hist.history['val_loss'])
    print(f"Epoch {epoch+1} train time: {train_times[-1]:.2f} s")

# 6. Evaluation on test data
test_loss, test_acc = model.evaluate(X_test_flat, y_test_cat, batch_size=256, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# 7. Accuracy and loss curves
plt.figure(figsize=(8, 5))
plt.plot(history_acc, label='Train Accuracy')
plt.plot(history_val_acc, label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history_loss, label='Train Loss')
plt.plot(history_val_loss, label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.0, 1.2])
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 8. Confusion Matrix
y_pred = np.argmax(model.predict(X_test_flat, batch_size=256), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mnist_labels)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix – Test Data")
plt.tight_layout()
plt.show()

# 9. Classification Report
print("\nClassification Report (Precision, Recall, F1-Score):\n")
print(classification_report(y_test, y_pred, target_names=mnist_labels, digits=4))
