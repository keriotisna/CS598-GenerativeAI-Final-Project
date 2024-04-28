import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision
from torchvision.utils import make_grid
from torchvision import transforms

from datasets import *
from styleExtraction import *
from DDPM import *






# TODO: Add letterNum to select individual latters as well
def getAugmentedFontSamples(fontNum, letterNum, transform, numSamples, showPlots=True):
    
    
    dataPath = os.path.normpath(r'Data\FONTS\character_font.npz')
    loadedData = np.load(dataPath)

    images = loadedData['images']
    labels = loadedData['labels']

    baseSample = images[fontNum*26]
    sample_tensor = torch.from_numpy(baseSample).unsqueeze(0).to(torch.float32)

    if showPlots:
        plt.imshow(baseSample), plt.title(0), plt.show()

    newSamples = []

    for _ in range(500):
        transformed = transform(sample_tensor)
        newSamples.append(transformed)
        
    stackedSamples = torch.stack(newSamples, dim=0)

    if showPlots:
        grid = make_grid(stackedSamples, normalize=True, nrow=int(np.sqrt(stackedSamples.shape[0])))
        img = torchvision.transforms.ToPILImage()(grid)
        img.show()
    
    return stackedSamples












def main():
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # TODO: Try more transforms
    randomRotation = transforms.RandomApply([transforms.RandomRotation((-15, 15))])
    randomBlur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 0.6))], p=0.25)

    augmentationTransform = transforms.Compose([
        # randomRotation,
        randomBlur,
        transforms.RandomAffine(degrees=(0, 0), translate=(0.05, 0.1), scale=(0.9, 1.1), shear=(0, 1)),
        transforms.Resize((28, 28)),
    ])

    augmentedSamples = getAugmentedFontSamples(fontNum=0, letterNum=0, transform=augmentationTransform, numSamples=500, showPlots=True)

    
    numClusters = 8

    # Get a combined train and test set since validation is for losers
    trainset = getFullDataset()
    flattenedData = torch.flatten(augmentedSamples, start_dim=1).numpy()

    # Get formatted information about what data indices belong to what cluster
    CLUSTER_DATA, CLUSTERINGS, REDUCTIONS = getClusterDataForLetters(trainset=trainset, numClusters=numClusters, showPlots=False, targetClasses=[1], getMappingInfo=True)
    
    LETTER_ID = 1
    clustering = CLUSTERINGS[LETTER_ID]
    reduction = REDUCTIONS[LETTER_ID]

    reduced = reduction.transform(flattenedData)
    clusterLabels = clustering.predict(reduced)
    
    
    indices, clusterCounts = np.unique(clusterLabels, return_counts=True)
    mostCommonCluster = indices[np.argmax(clusterCounts)]
    print(f'Most common cluster: {mostCommonCluster}')
    
    
    # Create individual data subsets for each cluster
    subsets = createSubsets(trainset, CLUSTER_DATA)

    currentSubset = subsets[1]
    currentSubset: ClusterDataset
    
    # Add our "handwriting" samples to the best cluster and revisualize
    currentSubset.addSamplesToData(newFeatures=augmentedSamples, newLabels=torch.full((augmentedSamples.shape[0],), fill_value=mostCommonCluster))
    currentSubset.visualizeClusterDataset()
    
    
    
    
    # LOAD MODEL
    model = UNetDeep(numClasses=numClusters)
    betas = (1e-4, 0.02)
    numTimesteps = 500
    ddpm = DDPM(model=model, betas=betas, numTimesteps=numTimesteps, dropoutRate=0.4, device=device, numClasses=numClusters)
    ddpm.to(device)

    # Load individual models if we need to
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))
    
    # TODO: Try more transforms
    randomRotation = transforms.RandomApply([transforms.RandomRotation((-10, 10))])
    randomBlur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 0.6))], p=0.25)
    transform = transforms.Compose([
        # randomRotation,
        randomBlur,
        transforms.RandomAffine(degrees=(0, 0), translate=(0.05, 0.1), scale=(0.9, 1.1)),
        transforms.Resize((28, 28)),
    ])
    
    
    ddpm = None
    fineTuneDDPM(ddpm=ddpm, numClasses=numClusters, epochs=200, batch_size=256, numTimesteps=500, dataset=currentSubset, label='A-FineTuned-Deep', transform=transform)





if __name__ == '__main__':
    main()

