import torch
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd




class ClusterDataset(Dataset):
    
    """
    A dataset meant to hold a single letter from EMNIST with subcategories as stylistic clusters
    """
    
    def __init__(self, features, labels, transform=None):
        
        self.features = features
        self.labels = labels
        
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.features[idx, ...]), self.labels[idx]
        else:
            return self.features[idx, ...], self.labels[idx]



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
    


def createSubsets(dataset, clusterData, transform):
    
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
        
        clusterSet = ClusterDataset(features=ALL_DATA_COMBINED, labels=ALL_LABELS_COMBINED, transform=transform)
        
        subsets[label] = clusterSet
    
    return subsets


def getFullDataset():
    
    """
    Gets and combines the training, validation and test datasets into a single larger dataset that can be used
    
    We don't need training or validation sets and really just need the raw data
    """
    
    trainset = EMNISTDataset(EMNISTDataset._TRAIN_CSV)
    testset = EMNISTDataset(EMNISTDataset._TEST_CSV)

    traindata = trainset.images
    trainlabels = trainset.labels

    testdata = testset.images
    testlabels = testset.labels
    
    combinedImages = torch.concat((traindata, testdata), dim=0)
    combinedLabels = torch.concat((trainlabels, testlabels), dim=0)
    
    combinedDataset = EMNISTDataset(images=combinedImages, labels=combinedLabels)
    

    return combinedDataset