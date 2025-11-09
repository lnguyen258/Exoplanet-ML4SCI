# BYOL for Protoplanetary Disk Images

This repository contains an implementation of Bootstrap Your Own Latent (BYOL) self-supervised learning method ([see paper](https://arxiv.org/abs/2006.07733)), specifically adapted for grayscale protoplanetary disk images.

## BYOL Architecture and Mechanism

BYOL (Bootstrap Your Own Latent) is a self-supervised learning method that learns image representations without using negative pairs. The architecture consists of:

- Two neural networks: online and target networks
- Both networks have the same architecture but different weights
- Each network contains:
    - Encoder (ResNet backbone)
    - Projector (MLP)
    - The online network has an additional predictor network

![a simple architecture demonstration](asset/image_1.png)

The learning process:
1. Two random augmented views are created from one image
2. Online network processes first view
3. Target network processes second view
4. Model learns by predicting target network representations
5. Target network parameters updated through exponential moving average

## Dataset

The dataset consists of protoplanetary disk grayscale images processed from FITS (Flexible Image Transport System) files. 

### Data Setup

1. Download FITS files from [here](https://drive.google.com/drive/folders/1VkS3RHkAjiKjJ6DnZmEKZ_nUv4w6pz7P)
2. Create directory structure:
```
data/
    ├── train/
    └── val/
```
3. Place FITS files in respective directories

## Installation

```bash
# Clone repository
git clone https://github.com/lnguyen258/Exoplanet-ML4SCI.git
cd exoplanet-ml4sci

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Training

Basic training command:
```bash
python train.py --data_dir data/train
```

## Checkpoints
After training, a checkpoint dir will be generated in the executed folder

- Full model (for resuming training): `checkpoint/full_model/`
- Online encoder (for downstream tasks): `checkpoint/online_encoder/`

## Contributing

You can modify:
- Augmentation methods in `src/utils/augment.py`
- Dataset structure in `src/utils/dataset.py`
- Try different ResNet models in `train.py`

Note: This implementation is customized for grayscale images. Significant modifications are required for RGB input.