import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from collections import defaultdict

from datasets import EMNISTDataset



def getClusterDataForLetters(trainset:Dataset, numClusters=12, gridSize=5, numClasses=26, showPlots=False):


    """
    Gets a CLUSTER_DATA dictionary for usage in the 
    """

    # This is a dict which holds individual indices of data samples belonging to each cluster and is strudtured as follows:
    '''
    CLUSTER_DATA = {
        intLabel1: [
            [indicesForCluster1],
            [indicesForCluster2]
        ],
        intLabel2: [
            [indicesForCluster1],
            [indicesForCluster2]
        ],
        ...
    }
    '''
    CLUSTER_DATA = defaultdict(list)

    sampleCount = gridSize**2

    width = 2*numClusters
    if showPlots:
        fig, axs = plt.subplots(26, numClusters, figsize=(width, 2*width))
    
    for letterIdx in range(1, numClasses+1):
        # Filter images and labels for the current letter
        letterIndices = np.where(trainset.labels == letterIdx)[0]
        letterImages = trainset.images[letterIndices]
        
        # Flatten the images
        flatImages = torch.flatten(letterImages, start_dim=1).numpy()
        
        # Perform PCA
        # pca = PCA(n_components=50)
        # features = pca.fit_transform(flat_images)

        isomap = Isomap(n_components=20, n_jobs=-1, n_neighbors=20)
        features = isomap.fit_transform(flatImages)
        
        # Cluster the data points
        kmeans = KMeans(n_clusters=numClusters, random_state=42)
        clusterLabels = kmeans.fit_predict(features)

        # Really uneven clusters
        # spectralClusters = SpectralClustering(n_clusters=numClusters, random_state=42)
        # clusterLabels = spectralClusters.fit_predict(features)

        clusterPopulations = []
        
        # Display samples from each cluster in a 4x4 grid
        for cluster in range(numClusters):
            originalIndices = np.where(clusterLabels == cluster)[0]
            
            CLUSTER_DATA[letterIdx].append(letterIndices[originalIndices])
            clusterPopulations.append(originalIndices.shape[0])
            
            if not showPlots:
                continue
            
            clusterSamples = np.random.choice(originalIndices, size=sampleCount, replace=True)
            
            ax = axs[letterIdx-1, cluster]
            ax.axis('off')
            
            for i in range(gridSize):
                for j in range(gridSize):
                    sampleIndex = clusterSamples[i*gridSize+j]
                    image = letterImages[sampleIndex].squeeze()
                    ax_sub = ax.inset_axes([j*0.25, i*0.25, 0.25, 0.25])
                    ax_sub.imshow(image, cmap='gray')
                    ax_sub.axis('off')
            
            ax.set_title(f"Cluster {cluster+1}", fontsize=8)
            
        print(f'Letter {EMNISTDataset.intToStrDict[letterIdx]} cluster counts:\n{clusterPopulations}')

    if showPlots:
        plt.tight_layout()
        plt.show()
        
    return CLUSTER_DATA
