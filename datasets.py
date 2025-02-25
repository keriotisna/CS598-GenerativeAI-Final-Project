import PIL.Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd

import torchvision
from torchvision.utils import make_grid
import numpy as np
import PIL


class ClusterDataset(Dataset):
    
    """
    A dataset meant to hold a single letter from EMNIST with subcategories as stylistic clusters
    
    Arguments:
        features: A (N, 1, 28, 28) grayscale image tensor
        labels: A (N,) tensor holding style labels for each image
    """
    
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        
        # Features is (N, C, 28, 28)
        self.setFeaturesAndLabels(features, labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx, ...], self.labels[idx]
        
        
    def addSamplesToData(self, newFeatures:torch.Tensor, newLabels:torch.Tensor):
    
        """
        Adds new samples and their corresponding labels to the existing dataset.
        
        newFeatures should have shape (N, 1, 28, 28) for the EMNIST dataset
        newLabels should have shape (N,)
        """
    
        assert newFeatures.shape[0] == newLabels.shape[0]
    
        features, labels = self.getFeaturesAndLabels()
    
        newFeatures = torch.concat((features, newFeatures), dim=0)
        newLabels = torch.concat((labels, newLabels), dim=0)
    
        self.setFeaturesAndLabels(newFeatures, newLabels)
    
    
    def getFeaturesAndLabels(self):
        
        """
        Return features and labels for interaction on the CPU
        """
        
        return self.features.detach_().cpu(), self.labels.detach_().cpu()
    
    def setFeaturesAndLabels(self, features: torch.Tensor, labels: torch.Tensor):
        
        """
        Sets features and labels while also sending data back to GPU
        """
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.features = features.to(device)
        self.labels = labels.to(device)

    
    def getClusterVisualizations(self) -> PIL.Image.Image:
        
        """
        Visualize an entire ClusterDataset showing the individual samples in their clusters.
        """
        
        features, labels = self.getFeaturesAndLabels()
        
        # Generate an image displaying all clusters of letters
        clusterImages = []
        for clusterNum in range(len(torch.unique(labels))):
            currentFeatures = features[labels == clusterNum]
            grid = make_grid(currentFeatures, nrow=int(np.sqrt(currentFeatures.shape[0])), normalize=True)
            
            t = torchvision.transforms.Compose([
                torchvision.transforms.Resize((1024, 1024))
            ])
            
            # grid is (3, 28, 28) so average to get grayscale
            clusterImages.append(torch.mean(t(grid), dim=0))
            
        fullImage = torch.stack(clusterImages, dim=0).unsqueeze(1)
        newGrid = make_grid(fullImage, nrow=len(clusterImages)//2, normalize=True)
        img = torchvision.transforms.ToPILImage()(newGrid)
        # img.show()
        return img
    
    
    def normalizeData(self, transform:torchvision.transforms):
        
        """
        Normalize data in the dataset based on the provided transform
        """
        
        features, labels = self.getFeaturesAndLabels()
        
        # Transform all the features, then zero mean and unit variance
        features = transform(features)
        
        mean = torch.mean(features)
        std = torch.std(features)
        
        normFeatures = features - mean
        normFeatures = normFeatures/std
        
        self.setFeaturesAndLabels(normFeatures, labels)
        
        return mean, std



class EMNISTDataset(Dataset):
    
    """
    The EMNIST handwritten dataset sourced from:
    https://www.kaggle.com/datasets/crawford/emnist
    """
    
    # A dict to convert from integer labels to string representations
    intToStrDict = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}
    NUM_CLASSES = 26
    
    _TRAIN_CSV = 'Data\EMNIST\emnist-letters-train.csv'
    _TEST_CSV = 'Data\EMNIST\emnist-letters-test.csv'

    def __init__(self, csvFile=None, images=None, labels=None):
        
        if csvFile is not None:
            self.data = pd.read_csv(csvFile, header=None)
        
        if images is None:
            self.images = torch.Tensor(self.data.iloc[:, 1:].values.reshape(-1, 1, 28, 28, order='A')).to(torch.float32)
        else: self.images = images

        if labels is None:
            self.labels = torch.Tensor(self.data.iloc[:, 0].values).to(torch.int16)
        else:
            self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label
    





class EMNISTDatasetByClass(EMNISTDataset):
    
    """
    The EMNIST handwritten dataset sourced from:
    https://www.kaggle.com/datasets/crawford/emnist
    
    This is a larger class-unbalanced letter dataset
    """
    
    # A dict to convert from integer labels to string representations
    intToStrDict = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}
    
    _TRAIN_CSV = 'Data\EMNIST\emnist-byclass-train.csv'
    _TEST_CSV = 'Data\EMNIST\emnist-byclass-test.csv'
    
    def __init__(self, csvFile=None, images=None, labels=None):
        
        if csvFile is not None:
            self.data = pd.read_csv(csvFile, header=None)
        
        if images is None:
            self.images = torch.Tensor(self.data.iloc[:, 1:].values.reshape(-1, 1, 28, 28, order='A')).to(torch.float32)
        else: self.images = images

        if labels is None:
            self.labels = torch.Tensor(self.data.iloc[:, 0].values).to(torch.int16)
        else:
            self.labels = labels
            
        # Get only letter samples and indices, then reindex to start from 0 instead.
        letterIndices = np.arange(10, 62)
        
        selectedImages = []
        selectedLabels = []
        
        for validIdx in letterIndices:
            
            selectedImages.append(self.images[self.labels == validIdx])
            selectedLabels.append(self.labels[self.labels == validIdx])

        capitalImages = selectedImages[:26]
        capitalLabels = selectedLabels[:26]
        
        lowerImages = selectedImages[26:]
        lowerLabels = selectedLabels[26:]

        del selectedImages, selectedLabels
        
        allLetters = []
        allLabels = []
        
        for cI, cL, lI, lL in zip(capitalImages, capitalLabels, lowerImages, lowerLabels):
            
            combinedImages = torch.concat((cI, lI), dim=0)
            cL = cL - 9 # One index the capital labels since index of a=1
            lL = lL - (9+26) # One index the lowercase labels
            
            combinedLabels = torch.concat((cL, lL), dim=0)
            
            allLetters.append(combinedImages)
            allLabels.append(combinedLabels)

        self.images = torch.concat(allLetters, dim=0)
        self.labels = torch.concat(allLabels, dim=0)







def createSubsets(dataset, clusterData) -> dict[ClusterDataset]:
    
    """
    Breaks a formatted CLUSTER_DATA dictionary into separate datasets with new indices
    For each letter, we will get a new dataset with numClusters classes to mimic different handwriting styles
    """
    
    subsets = {}

    for label, originalIndices in clusterData.items():
        
        ALL_DATA = []
        ALL_LABELS = []
        
        for i, indices in enumerate(originalIndices):
            subset = Subset(dataset, indices)
            
            stackedLabels = torch.full((len(subset),), i)
            
            subsetLoader = DataLoader(subset)
            stackedFeatures = torch.concat([feats for feats, _ in subsetLoader], dim=0).clone().detach()
            
            ALL_DATA.append(stackedFeatures)
            ALL_LABELS.append(stackedLabels)

        ALL_DATA_COMBINED = torch.concat(ALL_DATA, dim=0)
        ALL_LABELS_COMBINED = torch.concat(ALL_LABELS, dim=0)
        
        clusterSet = ClusterDataset(features=ALL_DATA_COMBINED, labels=ALL_LABELS_COMBINED)
        
        subsets[label] = clusterSet
    
    return subsets


def getFullDataset(useUnbalanced=False):
    
    """
    Gets and combines the training, validation and test datasets into a single larger dataset that can be used
    
    We don't need training or validation sets and really just need the raw data
    """
    
    if not useUnbalanced:
        trainset = EMNISTDataset(EMNISTDataset._TRAIN_CSV)
        testset = EMNISTDataset(EMNISTDataset._TEST_CSV)
    else:
        trainset = EMNISTDatasetByClass(EMNISTDatasetByClass._TRAIN_CSV)
        testset = EMNISTDatasetByClass(EMNISTDatasetByClass._TEST_CSV)

    traindata = trainset.images
    trainlabels = trainset.labels

    testdata = testset.images
    testlabels = testset.labels
    
    combinedImages = torch.concat((traindata, testdata), dim=0)
    combinedLabels = torch.concat((trainlabels, testlabels), dim=0)
    
    combinedDataset = EMNISTDataset(images=combinedImages, labels=combinedLabels)
    

    return combinedDataset