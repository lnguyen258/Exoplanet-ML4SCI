import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.model import BYOL
from src.utils import DiskDataset, augment_compose

# Intialize ResNet50, modify first conv for grayscale input, modify cls head to pass 
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc = nn.Identity()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory' )
args = parser.parse_args()

BATCH_SIZE = 8
EPOCHS = 1000
LR = 2e-5
NUM_GPUS = 1
IMAGE_SIZE = 224
NUM_WORKERS = 8

# Set up data augmentations
augmentor = augment_compose(IMAGE_SIZE)

class RepresentationLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super(RepresentationLearner, self).__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, view1, view2):
        return self.learner(view1, view2)

    def training_step(self, images, _):
        view1, view2 = images
        loss = self.forward(view1, view2)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        return [optimizer], [scheduler]
    
    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()
    
if __name__ == '__main__':
    train_set = DiskDataset(args.data_dir, transform=augmentor)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = RepresentationLearner(
        resnet,
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99,
        use_momentum=True,
    )

    trainer = pl.Trainer(
        max_epochs = EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
    )

    trainer.fit(model, train_loader)
