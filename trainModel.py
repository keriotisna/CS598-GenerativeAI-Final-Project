import time
import torch

import torchvision
from torchvision.utils import make_grid
from torchvision import transforms

import numpy as np

from datasets import *
from styleExtraction import *
from DDPM import trainDDPM

torch.backends.cudnn.benchmark = True

# Kill debuggers for training
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

def main():

    numClusters = 12

    # Get a combined train and test set since validation is for losers
    trainset = getFullDataset()

    # Get formatted information about what data indices belong to what cluster
    CLUSTER_DATA = getClusterDataForLetters(trainset=trainset, numClusters=numClusters, showPlots=False, numClasses=1)

    # TODO: Try more transforms
    randomRotation = transforms.RandomApply([transforms.RandomRotation((-15, 15))])
    transform = transforms.Compose([
        # transforms.ToTensor()
        # transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomAdjustSharpness(2),
        randomRotation
    ])

    # Create individual data subsets for each cluster
    subsets = createSubsets(trainset, CLUSTER_DATA, transform=transform)
    letterASubset = subsets[1]

    # Generate an image displaying all clusters of letters
    clusterImages = []
    for clusterNum in range(len(CLUSTER_DATA[1])):
        currentFeatures = letterASubset.features[letterASubset.labels == clusterNum]
        grid = make_grid(currentFeatures, nrow=int(np.sqrt(currentFeatures.shape[0])), normalize=True)
        
        t = transforms.Compose([
            transforms.Resize((512, 512))
        ])
        
        # grid is (3, 28, 28) so average to get grayscale
        clusterImages.append(torch.mean(t(grid), dim=0))

    fullImage = torch.stack(clusterImages, dim=0).unsqueeze(1)
    newGrid = make_grid(fullImage, nrow=len(clusterImages)//2, normalize=True)
    img = torchvision.transforms.ToPILImage()(newGrid)
    img.show()

    # Start training for one cluster
    startTime = time.time()
    trainDDPM(numClasses=numClusters, epochs=1200, batch_size=512, numTimesteps=500, dataset=letterASubset)

    print(f'Training finished after {time.time()-startTime}')

if __name__ == '__main__':
    main()