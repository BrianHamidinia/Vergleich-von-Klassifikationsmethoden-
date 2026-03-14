

# Vergleich von datengetriebenen Klassifikationsmethoden

**Projekt:** Vergleich von datengetriebenen Klassifikationsmethoden auf MNIST und CIFAR-10  
**Autor:** Mohammad (Brayan) Hamidinia  
**Email:** brayanhamidinia@gmail.com


## Kurzbeschreibung
Dieses Projekt untersucht und vergleicht verschiedene datengetriebene Klassifikationsmethoden für Bilddaten. Dabei kommen **Multilayer Perceptron (MLP)**, **Convolutional Neural Networks (CNN)**, **Autoencoder** und **Generative Adversarial Networks (GANs)** zum Einsatz.  
Ein besonderes Augenmerk liegt auf der Analyse, wie durch GANs generierte künstliche Bilder die Klassifikationsqualität beeinflussen. Evaluierungen erfolgen mit den Benchmark-Datensätzen **MNIST**, und **CIFAR-10**.

## Methoden & Tools
- **MLP, CNN, Autoencoder, GAN**  
- Datenvorverarbeitung: Normalisierung, One-Hot-Encoding, Feature-Extraktion  
- Evaluierung: Accuracy, Confusion Matrix, MSE, SSIM, PSNR  
- **Python 3.10**, **TensorFlow 2.10.0**, **Visual Studio Code**  
- Hardware: NVIDIA GeForce GTX 1650, Windows 11

## Dateien & Downloads
Aufgrund der Dateigrößen (>100 MB) werden die großen Datendateien separat bereitgestellt:  

- [Große Dateien auf Google Drive](https://drive.google.com/drive/folders/1OGGuq5kiixYuVW6M1hu1WVoV_RVPgs_R?usp=drive_link)

## Fazit
CNNs liefern insbesondere bei komplexen Datensätzen wie CIFAR-10 die besten Klassifikationsergebnisse. Die Nutzung von GAN-generierten Bildern kann die Accuracy auf MNIST deutlich steigern. Autoencoder sind besonders nützlich zur Feature-Reduktion und Effizienzsteigerung, während MLPs bei komplexeren Datensätzen schnell an ihre Grenzen stoßen.
