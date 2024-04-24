from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np



NUM_CLASSES = 26


class BasicBlock(nn.Module):
    
    printOutSize = False
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, activation=nn.ReLU(), **kwargs):
        super(BasicBlock, self).__init__()
        
        self.activation = activation
        
        self.seq = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False, **kwargs),
                nn.BatchNorm2d(num_features=out_channels),
                self.activation,
            )
        
    def forward(self, x):

        out = self.seq(x)
        if self.printOutSize:
            print(f'{out.shape=}')
        return out


class LinearEmbedding(nn.Module):
    
    """
    A simple linear embedding layer which embeds things like class and diffusion timestep
    """
    
    def __init__(self, in_features, out_features, activation=nn.GELU()):
        super(LinearEmbedding, self).__init__()
        
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        
        # Don't use bias since we normalize it out
        self.embeddingNetwork = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=False),
            nn.LayerNorm(normalized_shape=out_features),
            self.activation,

            nn.Linear(in_features=out_features, out_features=out_features, bias=False),
            nn.LayerNorm(normalized_shape=out_features),
            self.activation,
            
            nn.Linear(in_features=out_features, out_features=out_features),
        )
        
        
    def forward(self, x):
        
        return self.embeddingNetwork(x)



class ResidualBlock3(nn.Module):
        
    def __init__(self, in_channels, out_channels, activation:nn.Module=nn.GELU(), kernel_size:int=3, 
                stride:int=1, padding:int='same'):
        super().__init__()
        
        
        self.activation = activation
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            self.activation,
        )
        
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            self.activation,
        )
        
        if in_channels != out_channels:
            self.residualLayer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.residualLayer = nn.Identity()
            
        self.outNorm = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, x):

        residual = self.residualLayer(x)

        y1 = self.c1(x)
        y = self.c2(y1)
        
        self.outsize = y.size()
        
        return self.outNorm(y + residual)



class UNetDown(nn.Module):
    
    """
    Take some input and downsample by a factor of 2
    """
    
    def __init__(self, in_channels, out_channels, activation=nn.GELU()):
        super(UNetDown, self).__init__()

        self.conv1 = nn.Sequential(
            ResidualBlock3(in_channels=in_channels, out_channels=out_channels, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.conv1(x)


class UNetUp(nn.Module):
    
    """
    Take an input and upsample by a factor of 2
    May need to adjust kernel size and stride to get it properly working
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=2, 
                stride=2, padding=1, activation=nn.GELU(), numPostResiduals=3):
        super(UNetUp, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            activation,
            *[ResidualBlock3(in_channels=out_channels, out_channels=out_channels, activation=activation) for _ in range(numPostResiduals)]
        )
        
    def forward(self, x, residualInput=None):
        
        conv1Out = self.conv1(x)
        # Concatenate results along channels dim
        if residualInput is not None:
            return torch.concat((conv1Out, residualInput), dim=1)
        else:
            return conv1Out



def getDDPMSchedule(startBeta: float, endBeta: float, timestepCount: int):
    
    # TODO: WHAT THE HELL DO EACH OF THESE DO
    
    # beta_t
    betas = torch.sqrt(torch.linspace(startBeta, endBeta, timestepCount))
    
    # alpha_t
    alphas = 1 - betas
    
    # alphabar_t
    alphaBar = torch.exp(torch.cumsum(torch.log(alphas), dim=0))
    
    # sqrtab
    sqrtAlphaBar = torch.sqrt(alphaBar)
    
    # sqrtmab
    sqrtComplimentAlphaBar = torch.sqrt(1 - alphaBar)
    
    # mab_over_sqrtmab
    betasOverSqrtComplimentAlphaBar = betas / sqrtComplimentAlphaBar
    
    return {
        'alphas': alphas,
        'oneOverSqrtAlphas': 1/torch.sqrt(alphas),
        'sqrtBetas': torch.sqrt(betas),
        'alphaBar': alphaBar,
        'sqrtAlphaBar': sqrtAlphaBar,
        'sqrtComplimentAlphaBar': sqrtComplimentAlphaBar,
        'betasOverSqrtComplimentAlphaBar': betasOverSqrtComplimentAlphaBar
    }
    


class DDPM(nn.Module):
    
    def __init__(self, model: nn.Module, betas: torch.Tensor, 
                numTimesteps: int, dropoutRate: float, device: str):
        super(DDPM, self).__init__()
        
        self.model = model
        self.betas = betas
        self.numTimesteps = numTimesteps
        self.dropoutRate = dropoutRate
        
        self.loss = nn.MSELoss()
        
        self.device = device
        
        # Save beta values in the model itself for access to them during the forward pass.
        for key, value in getDDPMSchedule(betas[0], betas[1], numTimesteps).items():
            self.register_buffer(key, value)
        
        
        
        pass
    
    def forward(self, x, classLabels):
        
        """
        Sample random timestep for random noise
        """
        
        randomTimes = torch.randint(1, self.numTimesteps+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)
        
        # This is the forward pass equation from the paper
        noisedSamples = self.sqrtAlphaBar.view(-1, 1, 1, 1)*x \
                        + self.sqrtComplimentAlphaBar.view(-1, 1, 1, 1)*noise
        
        classMask = torch.bernoulli(torch.zeros_like(classLabels)+self.dropoutRate).to(self.device)
        
        return self.loss(noise, self.model(noisedSamples, classLabels, randomTimes/self.numTimesteps, classMask))
    
    def sample(self, numSamples, size, classifierGuidance=0):
        
        # Start out our samples as pure noise
        noisySamples = torch.randn(numSamples, *size).to(self.device)
        
        classLabels = torch.arange(0, NUM_CLASSES).to(self.device)
        classLabels = classLabels.repeat(int(numSamples/classLabels.shape[0])) # TODO: WHAT THE FUCK IS THIS
        classLabels = classLabels.repeat(2)

        # Create a class mask of 0s in the first half and 1s in the second half.
        # We will use this for CFG where we want the model to learn without the class label
        classMask = torch.zeros(classLabels.shape).to(self.device)
        classMask = classMask.repeat(2)
        classMask[numSamples:] = 1
        
        x_i_store = [] # keep track of generated steps in case want to plot something 

        for time in range(self.numTimesteps, 0, -1):
            
            currentTimestep = torch.tensor([time/self.numTimesteps]).to(self.device).repeat(numSamples,1,1,1)
            
            noisySamples = noisySamples.repeat(2,1,1,1)
            currentTimestep = currentTimestep.repeat(2,1,1,1)
            
            # Create random noise for sample denoising
            z = torch.randn(numSamples, *size).to(self.device) if time > 1 else 0

            predictedNoise = self.model(noisySamples, classLabels, currentTimestep, classMask)
            
            # Use classifier free guidance by shifting predictions in the directions of 
            # guided samples
            guidedNoisePredictions = predictedNoise[:numSamples]
            unguidedNoisePredictions = predictedNoise[numSamples:]
            
            predictedNoise = (1+classifierGuidance)*guidedNoisePredictions - classifierGuidance*unguidedNoisePredictions
            
            # Pick out the guided noisy samples
            noisySamples = noisySamples[:numSamples]
            noisySamples = self.oneOverSqrtAlphas[time]*(noisySamples - predictedNoise*self.betasOverSqrtComplimentAlphaBar[time]) \
                            + self.sqrtBetas[time]*z
            
            
        return noisySamples, x_i_store




class UNet(nn.Module):
    
    def __init__(self, numClasses=NUM_CLASSES):
        super(UNet, self).__init__()
        
        
        self.numClasses = numClasses
        FC = 128 # Feature count
        self.FC = FC
        
        self.conv1 = BasicBlock(in_channels=1, out_channels=FC)
        self.down1 = nn.Sequential(
            UNetDown(in_channels=FC, out_channels=FC),
            ResidualBlock3(in_channels=FC, out_channels=FC),
            ResidualBlock3(in_channels=FC, out_channels=FC),
            ResidualBlock3(in_channels=FC, out_channels=FC),
        )
        
        self.down2 = nn.Sequential(
            UNetDown(in_channels=FC, out_channels=FC),
            ResidualBlock3(in_channels=FC, out_channels=2*FC),
            ResidualBlock3(in_channels=2*FC, out_channels=2*FC),
            ResidualBlock3(in_channels=2*FC, out_channels=2*FC),
        )
        
        self.vectorize = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(),
            # nn.LayerNorm(normalized_shape=2*FC),
            nn.GELU()
        )
        
        self.timeEncoder1 = LinearEmbedding(in_features=1, out_features=2*FC)
        self.classEncoder1 = LinearEmbedding(in_features=numClasses, out_features=2*FC)

        self.timeEncoder2 = LinearEmbedding(in_features=1, out_features=FC)
        self.classEncoder2 = LinearEmbedding(in_features=numClasses, out_features=FC)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2*FC, out_channels=2*FC, kernel_size=7, stride=7),
            nn.GroupNorm(num_groups=8, num_channels=2*FC),
            nn.GELU(),
        )
        
        self.up1 = UNetUp(in_channels=4*FC, out_channels=FC)
        self.up2 = UNetUp(in_channels=2*FC, out_channels=FC)
        
        self.consolidate = nn.Sequential(
            nn.Conv2d(in_channels=2*FC, out_channels=FC, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=FC),
            nn.ReLU(),
            nn.Conv2d(in_channels=FC, out_channels=1, kernel_size=3, stride=1, padding=1),
        )



    
    def forward(self, x:torch.Tensor, classLabels:torch.Tensor, 
                timesteps:torch.Tensor, classMask:torch.Tensor):
        
        oneHotClass = F.one_hot(classLabels, num_classes=self.numClasses).to(torch.float32)
        
        classMask = classMask[:, None]
        classMask = classMask.repeat(1, self.numClasses)
        classMask = (-1*(1-classMask)) # Flip zeroes and 1s
        
        oneHotClass = oneHotClass * classMask
        
        classEmbedding1 = self.classEncoder1(oneHotClass).view(-1, self.FC*2, 1, 1)
        classEmbedding2 = self.classEncoder2(oneHotClass).view(-1, self.FC, 1, 1)
        
        print(f'{classEmbedding1.shape=}')
        print(f'{classEmbedding2.shape=}')

        timeEmbedding1 = self.timeEncoder1(timesteps).view(-1, self.FC*2, 1, 1)
        timeEmbedding2 = self.timeEncoder2(timesteps).view(-1, self.FC, 1, 1)
        
        print(f'{timeEmbedding1.shape=}')
        print(f'{timeEmbedding2.shape=}')
        
        conv1Out = self.conv1(x)
        down1Out = self.down1(conv1Out)
        down2Out = self.down2(down1Out)
        
        vectorImage = self.vectorize(down2Out)
        
        up0Out = self.up0(vectorImage)
        up1Out = self.up1(classEmbedding1*up0Out+timeEmbedding1, down2Out)
        up2Out = self.up2(classEmbedding2*up1Out+timeEmbedding2, down1Out)
        
        out = self.consolidate(torch.cat((up2Out, x), 1))
        
        return out
        
    
    
    





def main():
    
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    BATCH_SIZE = 16
    INPUT_SIZE = (BATCH_SIZE, 1, 28, 28)

    randomLabels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,1)).to(device)
    randomTimes = torch.randint(0, 100, (BATCH_SIZE,1)).to(device).to(torch.float32)
    classMasks = torch.ones((BATCH_SIZE,)).to(device)

    dummyInput = torch.rand(INPUT_SIZE).to(device)


    module = UNet(numClasses=26)

    # profileModel(module, input_size=INPUT_SIZE)

    # Create an instance of the nn.module class defined above:
    module = module.to(device)

    output = module.forward(dummyInput, randomLabels, randomTimes, classMasks)
    print(output.shape)
    
    
    
    
    
    pass
















if __name__ == '__main__':
    main()