import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import imageio
import os
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 1. Load the MNIST dataset
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# 2. Split 10% of train for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.1,
    stratify=y_train_full  
)

# 3. Reshape for CNN (add channel dimension) and normalize
X_train_cnn = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_val_cnn   = X_val.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test_cnn  = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 4. One-hot encoding of labels
y_train_cat = to_categorical(y_train, 10)
y_val_cat   = to_categorical(y_val, 10)
y_test_cat  = to_categorical(y_test, 10)


# 3. Deep CNN model architecture / Tiefe CNN Modellarchitektur
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.35),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.4),
    
    GlobalAveragePooling2D(),  
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Training for 50 epochs / Training mit 50 Epochen
EPOCHS = 120
history = model.fit(
    X_train_cnn, y_train_cat,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=128,
    verbose=2
)

# 5. Evaluate on test data / Evaluation auf Testdaten
test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# 6a. Accuracy curve / Accuracy-Kurve
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy / Trainingsgenauigkeit')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy / Validierungsgenauigkeit')
plt.title('Accuracy per Epoch / Genauigkeit pro Epoche')
plt.xlabel('Epoch / Epoche')
plt.ylabel('Accuracy / Genauigkeit')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("acc_final.png")
plt.show()

# 6b. Loss curve / Loss-Kurve
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss / Trainingsverlust')
plt.plot(history.history['val_loss'], label='Validation Loss / Validierungsverlust')
plt.title('Loss per Epoch / Verlust pro Epoche')
plt.xlabel('Epoch / Epoche')
plt.ylabel('Loss / Verlust')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("loss_final.png")
plt.show()

# 6c. GIF for accuracy and loss / GIF für Accuracy und Loss
acc_pngs, loss_pngs = [], []
os.makedirs("frames_acc", exist_ok=True)
os.makedirs("frames_loss", exist_ok=True)
for i in range(1, len(history.history['accuracy'])+1):
    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(history.history['accuracy'][:i], label='Train Accuracy / Training')
    plt.plot(history.history['val_accuracy'][:i], label='Validation Accuracy / Validierung')
    plt.xlim([0, EPOCHS-1])
    plt.ylim([0.8, 1.0])
    plt.xlabel('Epoch / Epoche')
    plt.ylabel('Accuracy / Genauigkeit')
    plt.title(f'Accuracy up to Epoch {i} / Genauigkeit bis Epoche {i}')
    plt.legend()
    plt.tight_layout()
    fname_acc = f'frames_acc/acc_epoch_{i:02d}.png'
    plt.savefig(fname_acc)
    acc_pngs.append(fname_acc)
    plt.close()
    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'][:i], label='Train Loss / Training')
    plt.plot(history.history['val_loss'][:i], label='Validation Loss / Validierung')
    plt.xlim([0, EPOCHS-1])
    plt.ylim([0, max(max(history.history['loss']), max(history.history['val_loss']))])
    plt.xlabel('Epoch / Epoche')
    plt.ylabel('Loss / Verlust')
    plt.title(f'Loss up to Epoch {i} / Verlust bis Epoche {i}')
    plt.legend()
    plt.tight_layout()
    fname_loss = f'frames_loss/loss_epoch_{i:02d}.png'
    plt.savefig(fname_loss)
    loss_pngs.append(fname_loss)
    plt.close()

with imageio.get_writer('accuracy_progress.gif', mode='I', duration=0.7) as writer:
    for filename in acc_pngs:
        image = imageio.imread(filename)
        writer.append_data(image)
print("GIF for Accuracy saved: accuracy_progress.gif / GIF für Accuracy gespeichert.")

with imageio.get_writer('loss_progress.gif', mode='I', duration=0.7) as writer:
    for filename in loss_pngs:
        image = imageio.imread(filename)
        writer.append_data(image)
print("GIF for Loss saved: loss_progress.gif / GIF für Loss gespeichert.")

# 7. Confusion Matrix (image) / Confusion Matrix (Bild)
y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix – Test Data / Testdaten")
plt.show()

# 8. Precision, Recall, F1-Score (tabular) / Precision, Recall, F1-Score (Tabellarisch)
print("Classification Report (Precision, Recall, F1-Score):\n")
print(classification_report(y_test, y_pred, digits=4))

# 9. AUC-ROC per class / AUC-ROC pro Klasse
y_test_bin = to_categorical(y_test, num_classes=10)
y_score = model.predict(X_test_cnn)
auc_roc = {}
for i in range(10):
    auc = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    auc_roc[i] = auc
print("\nAUC-ROC for each class / AUC-ROC für jede Klasse:")
for i in range(10):
    print(f"Class {i} / Klasse {i}: {auc_roc[i]:.4f}")

# Optional: ROC curves for classes 0, 1, 2 / ROC-Kurven für 3 Klasse (z.B. 0,1,2)
plt.figure(figsize=(7, 6))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc_roc[i]:.3f}) / Klasse {i} (AUC={auc_roc[i]:.3f})")
plt.plot([0,1],[0,1],'k--',label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (some classes) / ROC Curve (einige Klassen)')
plt.legend()
plt.grid()
plt.show()

# 10. Sample images with prediction and groundtruth (2x10 grid)
# Beispielbilder mit Vorhersage und Groundtruth (2x10 Tabelle)
idxs = np.random.choice(len(X_test_cnn), 20, replace=False)
plt.figure(figsize=(20,5))
for i, idx in enumerate(idxs):
    plt.subplot(2, 10, i+1)
    plt.imshow(X_test_cnn[idx].reshape(28,28), cmap='gray')
    plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}")
    plt.axis('off')
plt.suptitle("Sample Images: True and Predicted / Beispielbilder: Wahrer und vorhergesagter Wert")
plt.tight_layout()
plt.show()




