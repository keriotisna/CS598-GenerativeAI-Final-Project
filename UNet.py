import torch
import torch.nn as nn
import torch.nn.functional as F


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



class UNetDownsample(nn.Module):
    
    """
    Take some input and downsample by a factor of 2
    """
    
    def __init__(self, in_channels, out_channels, activation=nn.GELU()):
        super(UNetDownsample, self).__init__()

        self.conv1 = nn.Sequential(
            ResidualBlock3(in_channels=in_channels, out_channels=out_channels, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.conv1(x)


class UNetUpsample(nn.Module):
    
    """
    Take an input and residual features, concatenate, then upsample by a factor of 2
    
    Arguments:
        input_channels: The number of channels in the input feature map
        residual_channels: The number of channels in the residual feature map
        out_channels: How many channels in the output
        includeEmbeddings: Whether or not to include time and class embeddings for this layer, just keep as True
    """
    
    def __init__(self, input_channels: int, residual_channels: int, out_channels: int, numClasses: int, kernel_size=2, 
                stride=2, padding=0, activation=nn.GELU(), numPostResiduals=3,
                includeEmbeddings=True):
        super(UNetUpsample, self).__init__()
        
        self.input_channels = input_channels
        self.residual_channels = residual_channels
        self.out_channels = out_channels
        self.includeEmbeddings = includeEmbeddings
        
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_channels+residual_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            activation,
            *[ResidualBlock3(in_channels=out_channels, out_channels=out_channels, activation=activation) for _ in range(numPostResiduals)]
        )
        
        if includeEmbeddings:
            self.timeEncoder = LinearEmbedding(in_features=1, out_features=input_channels)
            self.classEncoder = LinearEmbedding(in_features=numClasses, out_features=input_channels)
            
        
    def forward(self, x, residualInput, times, classLabels):
        
        """
        Performs a UNet upsampling forward pass. Concatenates residual input to x, then upsamples
        
        Arguments:
            x: (B, C1, N, N) tensor of inputs
            residualInput: (B, C2, N, N) tensor of previous activations
            times: (B,) tensor of integer timesteps
            classLabels: (B,) tensor of integer class labels
            
        Returns:
            conv1Out: (B, out_channels, 2N, 2N)
        """
        
        if self.includeEmbeddings and times is not None and classLabels is not None:
            timeEmbedding = self.timeEncoder(times.view(-1, 1)).view(-1, x.shape[1], 1, 1)
            classEmbedding = self.classEncoder(classLabels).view(-1, x.shape[1], 1, 1)
        
            x = x*classEmbedding + timeEmbedding
        
        x = torch.concat((x, residualInput), dim=1)
        
        conv1Out = self.conv1(x)
        
        return conv1Out


class UNet(nn.Module):
    
    def __init__(self, numClasses=10):
        super(UNet, self).__init__()
        
        
        self.numClasses = numClasses
        
        self.conv1 = BasicBlock(in_channels=1, out_channels=128)
        
        # Define down and up blocks next to each other so we can track channels easier
        self.down1 = nn.Sequential(
            UNetDownsample(in_channels=128, out_channels=128),
            ResidualBlock3(in_channels=128, out_channels=128),
            ResidualBlock3(in_channels=128, out_channels=128),
            ResidualBlock3(in_channels=128, out_channels=128),
        )
        self.up1 = UNetUpsample(input_channels=256, residual_channels=128, out_channels=128, includeEmbeddings=True, numClasses=numClasses)


        self.down2 = nn.Sequential(
            UNetDownsample(in_channels=128, out_channels=128),
            ResidualBlock3(in_channels=128, out_channels=256),
            ResidualBlock3(in_channels=256, out_channels=256),
            ResidualBlock3(in_channels=256, out_channels=256),
        )
        self.up2 = UNetUpsample(input_channels=256, residual_channels=256, out_channels=256, includeEmbeddings=True, numClasses=numClasses)

        
        # TODO: Try some normalization here, can't get the shapes to align right now
        self.vectorize = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(),
            # nn.LayerNorm(normalized_shape=2*FC),
            nn.GELU()
        )

        # This is just decoding from vector images to a new spatial representation, no conditioning or anything
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=7, stride=7),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.GELU(),
        )
        

        self.consolidate = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
        )



    
    def forward(self, x:torch.Tensor, classLabels:torch.Tensor, 
                timesteps:torch.Tensor, classMask:torch.Tensor):
        
        """
        classLabels: (B,) tensor of one-hot class labels
        timesteps: (B,) tensor of integer? timesteps
        classMask: (B,) tensor of integers
        """
        
        oneHotClass = F.one_hot(classLabels, num_classes=self.numClasses)
        
        classMask = classMask[:, None].to(torch.bool)
        classMask = ~classMask
        classMask = classMask.repeat(1, self.numClasses).to(torch.float32)
        
        oneHotClass = oneHotClass * classMask
        
        # Output is (B, 128, 28, 28)
        conv1Out = self.conv1(x)
        
        # Output is (B, 128, 14, 14)
        down1Out = self.down1(conv1Out)
        
        # Output is (B, 256, 7, 7)
        down2Out = self.down2(down1Out)
        
        # Output is (B, 256, 1, 1)
        vectorImage = self.vectorize(down2Out)
        
        # Output is (B, 256, 7, 7)
        up0Out = self.up0.forward(vectorImage)
        # print(f'{up0Out.shape=}')

        # Output is (B, 256, 14, 14)
        up2Out = self.up2.forward(up0Out, down2Out, timesteps, oneHotClass)
        # print(f'{up2Out.shape=}')
        
        # Output is (B, 128, 28, 28)
        up1Out = self.up1.forward(up2Out, down1Out, timesteps, oneHotClass)
        # print(f'{up1Out.shape=}')
        
        # Output is (B, 1, 28, 28))
        out = self.consolidate(up1Out)
        # print(f'{out.shape=}')

        return out