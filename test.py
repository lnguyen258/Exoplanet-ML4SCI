import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from torch.utils.data import DataLoader

from src.utils import DiskImageSet

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to trained encoder')
parser.add_argument('--data_dir', type=str, required=True, help='Path to raw data')
parser.add_argument('--n_clusters', type=int, required=True, help='Number of clusters')
args = parser.parse_args()

# Initiate encoder 
encoder = resnet50(weights=None)
encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
encoder.fc = nn.Identity()

if __name__ == '__main__':
    dataset = DiskImageSet(args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Load trained encoder
    encoder.load_state_dict(torch.load(args.checkpoint_dir))
    encoder.eval()

    all = []
    with torch.no_grad():
        for X in loader:
            rep = encoder(X)
            all.append(rep.cpu().numpy())
    
    all = np.concatenate(all, axis=0)
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all)

    # Save labels
    save_dir = 'log/labels/cluster_labels.npy'
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    np.save(save_dir, cluster_labels)

    # Visualize in 3D
    pca = PCA(n_components=3)
    features_3d = pca.fit_transform(all)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2],
                        c=cluster_labels, cmap='tab10', alpha=0.6, s=10)

    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_zlabel('PCA Component 3', fontsize=12)
    ax.set_title(f'K-Means Clustering (k={args.n_clusters})', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.8, pad=0.1)

    # Adjust viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    fig_dir = 'log/outputs/clustering_viz_3d.png'
    os.makedirs(os.path.dirname(fig_dir), exist_ok=True)
    plt.savefig(fig_dir, dpi=300, bbox_inches='tight')
    print("Saved clustering_viz_3d.png")




        
