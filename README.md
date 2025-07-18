# Monet-to-Photo Style Transfer with CycleGAN 🎨📸

This project implements a CycleGAN model to perform style transfer between Monet paintings and real photographs. The model learns to transform photos into the style of Monet paintings and vice versa, using unpaired image data. It leverages cycle-consistency loss, adversarial loss, and perceptual loss to ensure high-quality image translations. 🚀

## Table of Contents 📋
- [Overview](#overview) 🌟
- [Features](#features) ✨
- [Requirements](#requirements) 🛠️
- [Dataset](#dataset) 🖼️
- [Installation](#installation) 🔧
- [Usage](#usage) ▶️
- [Model Architecture](#model-architecture) 🏗️
- [Training](#training) 📈
- [Evaluation](#evaluation) 📊
- [Results](#results) 🥳
- [Contributing](#contributing) 🤝
- [License](#license) 📜

## Overview 🌟
CycleGAN is a powerful generative adversarial network that enables image-to-image translation without paired examples. This project uses a U-Net-based generator with ResNet blocks and a PatchGAN discriminator to achieve stunning style transfer between Monet paintings and photographs. The model is trained with adversarial, cycle-consistency, and perceptual losses to produce realistic and coherent outputs. 🎨

## Features ✨
- **CycleGAN Architecture**: Two generators (photo-to-Monet and Monet-to-photo) and two discriminators for robust translation. 🔄
- **Perceptual Loss**: Utilizes VGG16 features to enhance the visual quality of generated images. 🖌️
- **Image Buffer**: Prevents mode collapse by storing previously generated images. 🗄️
- **Learning Rate Scheduling**: Dynamically adjusts learning rates during training for better convergence. ⏳
- **Evaluation Metrics**: Computes FID and MiFID to assess the quality of generated images. 📏

## Requirements 🛠️
- Python 3.7+ 🐍
- PyTorch 🔥
- torchvision 📷
- NumPy 🔢
- Pillow 🖼️
- SciPy 🔬
- pandas 🐼
- tqdm ⏳

## Install the dependencies using:
```bash
pip install torch torchvision numpy pillow scipy pandas tqdm
```


## Dataset 🖼️
The dataset should include two directories:

photo/: Real photographs in .jpg format. 📸
monet/: Monet paintings in .jpg format. 🎨
eval/: Real Monet images for evaluation. 🖌️
Ensure all images are 256x256 pixels. You can use datasets like the Monet2Photo dataset from Kaggle. 📂

## Installation 🔧
Clone the repository:

```bash
git clone https://github.com/your-username/Monet-to-Photo-Style-Transfer-with-CycleGAN.git
cd Monet-to-Photo-Style-Transfer-with-CycleGAN

```
## Usage ▶️
Prepare the Dataset:
Place photo images in the photo/ directory. 📷
Place Monet paintings in the monet/ directory. 🖼️
Place evaluation images in the eval/ directory. 🖌️
Train the Model: Run the training script to train the CycleGAN model:
```bash
python main.py
```
This will:
### Train the model for 200 epochs. ⏳
Save model checkpoints every 20 epochs in the models/ directory. 💾
Save final generator models (G.pth and F.pth) in the Final_models/ directory. 🏁
Generate Monet-style images in the submission/ directory. 🖼️

### Evaluate the Model: Evaluate the generated images using FID and MiFID metrics:
```bash
python evaluation.py
```
The evaluation script will:
Load real Monet images from eval/ and generated images from submission/. 📂
Compute FID and MiFID scores. 📊
Save results to submission.csv. 📄


## Model Architecture 🏗️
**Generator**: A U-Net with 9 ResNet blocks, including convolutional layers, instance normalization, and ReLU activations. 🧠
**Discriminator**: A PatchGAN with multiple convolutional layers and LeakyReLU activations. ⚖️
**Loss Functions**:
Adversarial Loss (MSE): Ensures generated images are indistinguishable from real images. 🎯
Cycle-Consistency Loss (L1): Ensures images can be reconstructed after translation. 🔄
Perceptual Loss (L1): Uses VGG16 features to improve visual quality. 🖌️

## Training 📈
Hyperparameters:
Image Size: 256x256 🖼️
Batch Size: 3 📦
Learning Rate: 0.0002 🚀
Beta1: 0.5 ⚖️
Epochs: 200 ⏳
Cycle Loss Weight: 10.0 🔄
Perceptual Loss Weight: 5.0 🖌️
Training Process (in main.py):
Alternates between training generators and discriminators. 🔄
Uses Adam optimizers with learning rate scheduling (decays every 100 epochs). 📉
Saves checkpoints every 20 epochs and final models at the end. 💾

Evaluation 📊
Metrics (in evaluation.py):
FID: Measures similarity between feature distributions of real and generated images using InceptionV3. 📏
MiFID: Measures memorization by computing the minimum cosine distance between generated and real image features. 🔍
Process:
Subsampling ensures equal numbers of real and generated images. ⚖️
Processes images in batches (default batch size: 64). 📦
Saves results to submission.csv. 📄

## Performance Metrics


| Metric       | Value   | Training Details          |
|--------------|---------|---------------------------|
| FID Score    | 46.22   |                           |
| MiFID Score  | 0.222   | Trained for 200 epochs    |
|              |         | Image size: 256×256 px    |

## 📌 License

This project is for educational and research use only.
