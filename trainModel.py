import time
import torch

import torchvision
from torchvision.utils import make_grid
from torchvision import transforms

import numpy as np

from datasets import *
from styleExtraction import *
from DDPM import trainDDPM
from UNet import *

# Run a short benchmark before training to use the fastest convolution algorithm
torch.backends.cudnn.benchmark = True

# Kill debuggers for training
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


numClusters = 8

MODEL_BATCHES = [
    # {
    #     'model': UNetTiny(numClasses=numClusters),
    #     'batch_size': 1024,
    #     'modelName': 'Tiny',
    # },
    
    # {
    #     'model': UNetMedium(numClasses=numClusters),
    #     'batch_size': 512,
    #     'modelName': 'Medium',
    # },
    
    # {
    #     'model': UNetLarge(numClasses=numClusters),
    #     'batch_size': 512,
    #     'modelName': 'Large',
    # }
    
    {
        'model': UNetDeep(numClasses=numClusters),
        'batch_size': 128,
        'modelName': 'Deep',
    }
]



def main():

    
    # Get a combined train and test set since validation is for losers
    trainset = getFullDataset()

    targetClasses = np.arange(start=1, stop=2)

    # Get formatted information about what data indices belong to what cluster
    CLUSTER_DATA = getClusterDataForLetters(trainset=trainset, numClusters=numClusters, showPlots=False, targetClasses=targetClasses)
    
    # Create individual data subsets for each cluster
    subsets = createSubsets(trainset, CLUSTER_DATA)
    
    
    for params in MODEL_BATCHES:
    
        model = params['model']
        batch_size = params['batch_size']
        modelName = params['modelName']
    
        print(f'Training model {modelName}')
    
        validateModelIO(model=model, numClasses=numClusters)

        for letterNumber in subsets.keys():
            
            modelLabel = f'{modelName}-{EMNISTDataset.intToStrDict[letterNumber]}'
            
            letterSubset: ClusterDataset
            letterSubset = subsets[letterNumber]
            
            # TODO: Try more transforms
            randomRotation = transforms.RandomApply([transforms.RandomRotation((-10, 10))])
            randomBlur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 0.6))], p=0.25)
            transform = transforms.Compose([
                randomRotation,
                randomBlur,
                transforms.RandomAffine(degrees=(0, 0), translate=(0.05, 0.1), scale=(0.9, 1.1)),
                transforms.Resize((28, 28)),
            ])
            
            letterSubset.normalizeData(transform)
            letterSubset.visualizeClusterDataset()

            # Start training for one cluster
            startTime = time.time()
            
            trainDDPM(model=model, numClasses=numClusters, epochs=800, batch_size=batch_size, numTimesteps=500, dataset=letterSubset, label=modelLabel, transform=transform)

            print(f'Training finished after {time.time()-startTime}')

if __name__ == '__main__':
    main()