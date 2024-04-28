import os
import torch
import torch.nn as nn

import torch.utils
from torch.utils.data import DataLoader, Dataset

from torchvision.utils import save_image, make_grid
from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision import transforms

from UNet import *



class DDPM(nn.Module):
    
    def __init__(self, model: nn.Module, betas: torch.Tensor, 
                numTimesteps: int, dropoutRate: float, device: str, numClasses: int):
        super(DDPM, self).__init__()
        
        self.model = model.to(device)
        self.betas = betas
        self.numTimesteps = numTimesteps
        self.dropoutRate = dropoutRate
        self.numClasses = numClasses
        
        # Define loss in initialization since we don't want to define it for each forward pass
        self.loss = nn.MSELoss()
        self.device = device
        
        # betas are linearly interpolated from the start to the ending beta for each timestep
        # These represent the variance schedule which defines how much noise there is at each timestep
        betas = torch.linspace(betas[0], betas[1], numTimesteps+1)
        sqrtBetas = torch.sqrt(betas)

        # Formula is at the bottom of page 2 in the paper
        # \alpha = 1-\beta_t
        alphas = 1 - betas
        
        
        ###################################################
        # Training Equations
        ###################################################
        
        # This is the formula found at the bottom of page 2 where \bar{\alpha}_t is the cumulative product of previous \alpha values
        # \bar{alpha}_t = \prod^t_{s=1} \alpha_s
        alphaBar = torch.exp(torch.cumsum(torch.log(alphas), dim=0))
        sqrtComplimentAlphaBar = torch.sqrt(1 - alphaBar)
        
        
        ###################################################
        # Sampling Equations
        ###################################################
        
        # This is also in equation 11 and is used during sampling
        oneOverSqrtAlphas = 1/torch.sqrt(alphas)
        
        # This is used in equation 11 of the paper, it represents how much noise is in the current sample at a given timestep
        betasOverSqrtComplimentAlphaBar = betas / sqrtComplimentAlphaBar
        
        self.alphas = alphas.to(device)
        self.oneOverSqrtAlphas = oneOverSqrtAlphas.to(device)
        self.sqrtBetas = sqrtBetas.to(device)
        self.alphaBar = alphaBar.to(device)
        self.sqrtComplimentAlphaBar = sqrtComplimentAlphaBar.to(device)
        self.betasOverSqrtComplimentAlphaBar = betasOverSqrtComplimentAlphaBar.to(device)
        


    def forward(self, x: torch.Tensor, classLabels: torch.Tensor):
        
        """
        Sample random timestep for random noise
        """
        

        # Decide what samples should be masked out with 0s or 1s based on the dropout rate
        classMask = torch.bernoulli(torch.full(classLabels.shape, self.dropoutRate)).to(self.device)
        # Select random timesteps in the time range for the forward pass
        randomTimes = torch.randint(1, self.numTimesteps+1, (x.shape[0],)).to(self.device)
        
        # Random gaussian noise for reparameterization
        noise = torch.randn_like(x)
        
        # This is the forward pass equation from the paper
        noisedSamples = torch.sqrt(self.alphaBar[randomTimes, None, None, None])*x \
                        + self.sqrtComplimentAlphaBar[randomTimes, None, None, None]*noise
                        
        # Loss is MSE between true noise and model predicted noise
        return self.loss(noise, self.model(noisedSamples, classLabels, randomTimes/self.numTimesteps, classMask))
    
    
    
    def sample(self, numSamples, sampleSize, classifierGuidance=0.5, classLabels:torch.Tensor=None):
        
        if classLabels is None:
            # NOTE: Changed from 0 to self.numClasses
            classLabels = torch.arange(0, self.numClasses).to(self.device)
        
        # Define the shape of noisy samples to start the denoising process
        noiseShape = (numSamples, *sampleSize)
        
        # Start out our samples as pure noise
        noisySamples = torch.randn(*noiseShape).to(self.device)
        
        
        classLabels = classLabels.repeat(int(numSamples/classLabels.shape[0])) # TODO: WHAT THE FUCK IS THIS
        classLabels = classLabels.repeat(2)
        
        # Create a class mask of 0s in the first half and 1s in the second half.
        # We will use this for CFG where we want the model to learn without the class label
        classMask = torch.zeros(classLabels.shape).to(self.device)
        classMask[numSamples:] = 1
        
        # Store intermediate samples for plotting
        sampleStorage = []
        for time in range(self.numTimesteps, 0, -1):
            
            currentTimestep = torch.tensor([time/self.numTimesteps]).repeat(numSamples,1,1,1).to(self.device)
            
            noisySamples = noisySamples.repeat(2,1,1,1)
            currentTimestep = currentTimestep.repeat(2,1,1,1)
            
            # Get model predictions for noise when given class labels and times
            predictedNoise = self.model(noisySamples, classLabels, currentTimestep, classMask)
            
            # Use classifier free guidance by shifting predictions 
            # in the directions of guided samples
            unguidedNoisePredictions = predictedNoise[numSamples:]
            guidedNoisePredictions = predictedNoise[:numSamples]
            
            # We say the real predicted noise is noise pushed in the direction of the guided samples as opposed to the
            # unguided predictions. This helps improve the performance of the diffusion process at the cost of potentially
            # generating worse samples if the guidance isn't tuned properly.
            predictedNoise = (1+classifierGuidance)*guidedNoisePredictions - classifierGuidance*unguidedNoisePredictions
            
            
            # Create random noise for sample denoising
            z = torch.randn(*noiseShape).to(self.device) if time > 1 else 0
            
            # Pick out the guided noisy samples and denoise them using predicted 
            # noise and the appropriate scaling from the DDPM beta scheduling
            noisySamples = noisySamples[:numSamples]
            denoisedSamples = noisySamples - predictedNoise*self.betasOverSqrtComplimentAlphaBar[time]
            noisySamples = self.oneOverSqrtAlphas[time]*denoisedSamples + self.sqrtBetas[time]*z
            
            # TODO: Normalize stored samples? They appear very faded when looking at results
            if (time < 10) or (time % 20 == 0) or (time == self.numTimesteps):
                sampleStorage.append(noisySamples.detach().cpu().numpy())
            
        sampleStorage = np.array(sampleStorage)
        return noisySamples, sampleStorage






def trainDDPM(model, numClasses: int, epochs: int, batch_size: int, numTimesteps: int, dataset: Dataset, label: str, transform=None):
    
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Enable or disable automatic mixed precision for faster training
    USE_AMP = True
    lr = 1e-4
    save_model = True
    
    savePath = f'./DiffusionData/{label}/'
    
    if not os.path.isdir(savePath):
        os.mkdir(savePath)
    
    # CFG Guidance strength
    guidanceStrengths = [0, 1, 2]
    
    # betas was (1e-4, 0.02)
    # (1e-5, 0.04) and dropout 0.5 looked pretty noisy
    betas = (1e-4, 0.02)
    ddpm = DDPM(model=model, betas=betas, numTimesteps=numTimesteps, dropoutRate=0.4, device=device, numClasses=numClasses)
    ddpm.to(device)

    # Load individual models if we need to
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    # TODO: Try different optimizers
    # NOTE: DO NOT CHANGE num_workers>0 OR THE MODEL WON'T FUCKING TRAIN!!!!. WHY THE FUCK DOES THIS HAPPEN???!??!?!??!?!??!?!??!??!?!!!??!??!?!?
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False) # With pin_memory=False, speed is 6.4s/it; With pin_memory=True, speed is about the same
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)
    lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, factor=0.75, patience=40, cooldown=40)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


    pbar = tqdm(range(epochs))
    for ep in pbar:
        
        ddpm.train()
        newLR = optim.param_groups[0]['lr']

        for features, labels in dataloader:
            
            # Apply transform manually on the GPU
            features = transform(features)
            labels = labels

            for param in ddpm.parameters():
                param.grad = None

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                loss = ddpm(features, labels)
            
            scaler.scale(loss).backward() # Do backpropagation on scaled loss from AMP
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=5)
            scaler.step(optim)
            scaler.update()
            
            pbar.set_description(f"loss: {loss.item():.4f}, lr: {newLR:.6f}")

        lrScheduler.step(loss.item())

        if (ep%200==0 or ep == int(epochs-1)) and ep != 0:
            ddpm.eval()
            with torch.no_grad():
                numRows = 4
                totalSamples = numRows * numClasses
                for w_i, w in enumerate(guidanceStrengths):
                    x_gen, x_gen_store = ddpm.sample(totalSamples, (1, 28, 28), classifierGuidance=w)

                    # Append real samples at the bottom
                    x_real = torch.Tensor(x_gen.shape).to(device)
                    for k in range(numClasses):
                        for j in range(int(totalSamples/numClasses)):
                            try: 
                                idx = torch.squeeze((labels == k).nonzero())[j]
                            except:
                                idx = 0
                            x_real[k + (j * numClasses)] = features[idx]

                    dataSuffix = f'{label}-ep-{ep}_w-{w}_nc-{numClasses}_ts-{numTimesteps}'
                    allSamples = torch.cat([x_gen, x_real])
                    grid = make_grid(-allSamples+1, nrow=numClasses)
                    imageName = f"{dataSuffix}.png"
                    save_image(grid, savePath + imageName)
                    print('Saved ' + savePath + imageName)

                    fig, axs = plt.subplots(nrows=int(totalSamples/numClasses), ncols=numClasses,sharex=True,sharey=True,figsize=(8,3))
                    
                    # TODO: Move this somewhere else, it's really weird to have it here
                    def animateGif(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(totalSamples/numClasses)):
                            for col in range(numClasses):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*numClasses)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    ani = FuncAnimation(fig, animateGif, fargs=[x_gen_store], interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    gifName = f"{dataSuffix}.gif"
                    ani.save(savePath + gifName, dpi=100, writer=PillowWriter(fps=5))
                    print('Saved ' + savePath + gifName)
                    plt.close('all')
                    print()
                    
        # optionally save model
        if save_model and ep == epochs-1:
            modelName = f"model-{dataSuffix}.pth"
            torch.save(ddpm.state_dict(), savePath + modelName)



def main():
    
    from torchvision.datasets import MNIST
    
    trainDDPM(numClasses=10, epochs=2, batch_size=128, numTimesteps=400, dataset=MNIST('./Data', transform=transforms.ToTensor()))
    
    
if __name__ == '__main__':
    main()


