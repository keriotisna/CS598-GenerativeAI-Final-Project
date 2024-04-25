
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
from torchvision import transforms


from datasets import *
from styleExtraction import *
from DDPM import trainDDPM



def main():

    numClusters = 12

    trainset = getFullDataset()

    CLUSTER_DATA = getClusterDataForLetters(trainset=trainset, numClusters=numClusters, showPlots=False, numClasses=1)

    # TODO: Try more transforms
    randomRotation = transforms.RandomApply([transforms.RandomRotation((-15, 15))])
    transform = transforms.Compose([
        # transforms.ToTensor()
        # transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomAdjustSharpness(2),
        randomRotation
    ])

    subsets = createSubsets(trainset, CLUSTER_DATA, transform=transform)
    letterASubset = subsets[1]

    clusterImages = []

    for clusterNum in range(len(CLUSTER_DATA[1])):
        currentFeatures = letterASubset.features[letterASubset.labels == clusterNum]
        grid = make_grid(currentFeatures, nrow=int(np.sqrt(currentFeatures.shape[0])), normalize=True)
        
        t = transforms.Compose([
            transforms.Resize((512, 512))
        ])
        clusterImages.append(torch.mean(t(grid), dim=0))


    fullImage = torch.stack(clusterImages, dim=0).unsqueeze(1)
    newGrid = make_grid(fullImage, nrow=len(clusterImages)//2, normalize=True)
    img = torchvision.transforms.ToPILImage()(newGrid)
    img.show()

    startTime = time.time()
    trainDDPM(numClasses=numClusters, epochs=1200, batch_size=512, numTimesteps=400, dataset=letterASubset)

    print(f'Training finished after {time.time()-startTime}')

if __name__ == '__main__':
    main()