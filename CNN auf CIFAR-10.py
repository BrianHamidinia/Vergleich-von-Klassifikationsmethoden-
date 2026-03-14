import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import imageio
import os
from sklearn.model_selection import train_test_split

def load_dataset():
    # Original split: 50k train, 10k test
    (X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
    # Split 10% of train as validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full
    )
    # One-hot encode labels
    y_train_cat = to_categorical(y_train)
    y_val_cat   = to_categorical(y_val)
    y_test_cat  = to_categorical(y_test)
    return X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, y_test

def prep_pixels(*arrays):
    return [arr.astype('float32') / 255.0 for arr in arrays]

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def run_test_harness():
    # Dataset
    X_train, y_train, X_val, y_val, X_test, y_test_cat, y_test_label = load_dataset()
    X_train, X_val, X_test = prep_pixels(X_train, X_val, X_test)
    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

    # Model
    model = define_model()
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    it_train = datagen.flow(X_train, y_train, batch_size=64)
    steps = int(X_train.shape[0] / 64)

    # Training with timing
    EPOCHS = 200
    history_acc, history_val_acc = [], []
    history_loss, history_val_loss = [], []
    train_times = []
    for epoch in range(EPOCHS):
        start_time = time.time()
        hist = model.fit(
            it_train, steps_per_epoch=steps, epochs=1,
            validation_data=(X_val, y_val), verbose=2
        )
        end_time = time.time()
        train_times.append(end_time - start_time)
        history_acc.extend(hist.history['accuracy'])
        history_val_acc.extend(hist.history['val_accuracy'])
        history_loss.extend(hist.history['loss'])
        history_val_loss.extend(hist.history['val_loss'])
        print(f"Epoch {epoch+1} time: {train_times[-1]:.2f}s")

    # Final Test Evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Training Time per Epoch (seconds): {[f'{t:.2f}' for t in train_times]}")

    # Save/Plot Accuracy curve (Train & Val)
    plt.figure(figsize=(8, 5))
    plt.plot(history_acc, label='Train Accuracy')
    plt.plot(history_val_acc, label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.2, 1.0])
    plt.xlim([0, EPOCHS-1])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("acc_final.png")
    plt.show()

    # Save/Plot Loss curve (Train & Val)
    plt.figure(figsize=(8, 5))
    plt.plot(history_loss, label='Train Loss')
    plt.plot(history_val_loss, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0.0, 2.5])
    plt.xlim([0, EPOCHS-1])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("loss_final.png")
    plt.show()

    # GIF: dynamic accuracy and loss
    acc_pngs, loss_pngs = [], []
    os.makedirs("frames_acc", exist_ok=True)
    os.makedirs("frames_loss", exist_ok=True)
    for i in range(1, EPOCHS+1):
        # Accuracy GIF
        plt.figure(figsize=(6,4))
        plt.plot(history_acc[:i], label='Train Accuracy')
        plt.plot(history_val_acc[:i], label='Validation Accuracy')
        plt.xlim([0, EPOCHS-1])
        plt.ylim([0.2, 1.0])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy up to Epoch {i}')
        plt.legend()
        plt.tight_layout()
        fname_acc = f'frames_acc/acc_epoch_{i:03d}.png'
        plt.savefig(fname_acc)
        acc_pngs.append(fname_acc)
        plt.close()
        # Loss GIF
        plt.figure(figsize=(6,4))
        plt.plot(history_loss[:i], label='Train Loss')
        plt.plot(history_val_loss[:i], label='Validation Loss')
        plt.xlim([0, EPOCHS-1])
        plt.ylim([0.0, 2.5])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss up to Epoch {i}')
        plt.legend()
        plt.tight_layout()
        fname_loss = f'frames_loss/loss_epoch_{i:03d}.png'
        plt.savefig(fname_loss)
        loss_pngs.append(fname_loss)
        plt.close()

    with imageio.get_writer('accuracy_progress.gif', mode='I', duration=0.7) as writer:
        for filename in acc_pngs:
            image = imageio.imread(filename)
            writer.append_data(image)
    print("GIF for Accuracy saved: accuracy_progress.gif")

    with imageio.get_writer('loss_progress.gif', mode='I', duration=0.7) as writer:
        for filename in loss_pngs:
            image = imageio.imread(filename)
            writer.append_data(image)
    print("GIF for Loss saved: loss_progress.gif")

    # Confusion Matrix
    y_pred = np.argmax(model.predict(X_test, batch_size=64), axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix – Test Data")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Classification Report
    print("\nClassification Report (Precision, Recall, F1-Score):\n")
    print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))

    # Sample images with prediction/correctness
    idxs = np.random.choice(len(X_test), 20, replace=False)
    plt.figure(figsize=(20, 5))
    for i, idx in enumerate(idxs):
        true_label = class_labels[y_true[idx]]
        pred_label = class_labels[y_pred[idx]]
        correct = "✔" if y_true[idx] == y_pred[idx] else "✘"
        plt.subplot(2, 10, i+1)
        plt.imshow(X_test[idx])
        plt.title(f"T:{true_label}\nP:{pred_label}\n{correct}", fontsize=8)
        plt.axis('off')
    plt.suptitle("20 Sample Images: Ground Truth (T), Prediction (P), Correct/Incorrect")
    plt.tight_layout()
    plt.savefig("sample_images.png")
    plt.show()

    # t-SNE Embedding of Last Dense Layer
    from keras import Model as KerasModel
    feature_extractor = KerasModel(inputs=model.input, outputs=model.layers[-2].output)
    features_test = feature_extractor.predict(X_test, batch_size=64)
    tsne = TSNE(n_components=2, random_state=0, verbose=1)
    X_2d = tsne.fit_transform(features_test)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=y_true, cmap='tab10', s=6)
    plt.colorbar(scatter, ticks=range(10), label="Class")
    plt.title("t-SNE: Last Hidden Layer Embedding (CIFAR-10 Deep CNN)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig("tsne.png")
    plt.show()

# Entry point
run_test_harness()
