from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from UNet import *




def getDDPMSchedule(startBeta: float, endBeta: float, timestepCount: int):
    
    # TODO: WHAT THE HELL DO EACH OF THESE DO
    
    # betas are linearly interpolated from the start to the ending beta for each timestep
    # These represent the variance schedule which defines how much noise there is at each timestep
    # TODO: MAKE LINSPACE WORK
    # betas = torch.sqrt(torch.linspace(startBeta, endBeta, timestepCount)) # DOESN'T WORK, SHAPE MISMATCH?
    betas = (endBeta - startBeta) * torch.arange(0, timestepCount + 1, dtype=torch.float32) / timestepCount + startBeta

    # Formula is at the bottom of page 2 in the paper
    # \alpha = 1-\beta_t
    alphas = 1 - betas
    
    # This is the formula found at the bottom of page 2 where \bar{\alpha}_t is the cumulative product of previous \alpha values
    # \bar{alpha}_t = \prod^t_{s=1} \alpha_s
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
                numTimesteps: int, dropoutRate: float, device: str, numClasses: int):
        super(DDPM, self).__init__()
        
        self.model = model.to(device)
        self.betas = betas
        self.numTimesteps = numTimesteps
        self.dropoutRate = dropoutRate
        self.numClasses = numClasses
        
        self.loss = nn.MSELoss()
        
        self.device = device
        
        # Save beta values in the model itself for access to them during the forward pass.
        for key, value in getDDPMSchedule(betas[0], betas[1], numTimesteps).items():
            self.register_buffer(key, value)
        
        
            
    def forward(self, x, classLabels):
        
        """
        Sample random timestep for random noise
        """
        
        randomTimes = torch.randint(1, self.numTimesteps+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)
        
        # This is the forward pass equation from the paper
        noisedSamples = self.sqrtAlphaBar[randomTimes, None, None, None]*x \
                        + self.sqrtComplimentAlphaBar[randomTimes, None, None, None]*noise
        
        classMask = torch.bernoulli(torch.zeros_like(classLabels)+self.dropoutRate).to(self.device)
        
        return self.loss(noise, self.model(noisedSamples, classLabels, randomTimes/self.numTimesteps, classMask))
    
    
    def sample(self, numSamples, size, classifierGuidance=0):
        
        # Start out our samples as pure noise
        noisySamples = torch.randn(numSamples, *size).to(self.device)
        
        classLabels = torch.arange(0, self.numClasses).to(self.device)
        classLabels = classLabels.repeat(int(numSamples/classLabels.shape[0])) # TODO: WHAT THE FUCK IS THIS
        classLabels = classLabels.repeat(2)

        # Create a class mask of 0s in the first half and 1s in the second half.
        # We will use this for CFG where we want the model to learn without the class label
        classMask = torch.zeros(classLabels.shape).to(self.device)
        # classMask = classMask.repeat(2)
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
            
            if time%20==0 or time==self.numTimesteps or time<8:
                x_i_store.append(noisySamples.detach().cpu().numpy())
            
        x_i_store = np.array(x_i_store)
        return noisySamples, x_i_store






def trainDDPM(numClasses, epochs, batch_size, numTimesteps, dataset):
    
    
    ######################################################################################################
    # Verify model shape
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    BATCH_SIZE = 16
    INPUT_SIZE = (BATCH_SIZE, 1, 28, 28)

    randomLabels = torch.randint(0, numClasses, (BATCH_SIZE,)).to(device)
    randomTimes = torch.randint(0, 100, (BATCH_SIZE,)).to(device).to(torch.float32)
    classMasks = torch.ones((BATCH_SIZE,)).to(device)

    dummyInput = torch.rand(INPUT_SIZE).to(device)

    module = UNet(numClasses=numClasses)

    # Create an instance of the nn.module class defined above:
    module = module.to(device)

    output = module.forward(dummyInput, randomLabels, randomTimes, classMasks)
    if output is not None:
        print(output.shape)
    ######################################################################################################
    
    
    # hardcoding these here
    lr = 1e-4
    save_model = True
    savePath = './DiffusionData/'
    guidanceStrengths = [0.0, 0.5, 2.0] # strength of generative guidance

    ddpm = DDPM(model=UNet(numClasses=numClasses), betas=(1e-4, 0.02), numTimesteps=numTimesteps, dropoutRate=0.4, device=device, numClasses=numClasses)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    # TODO: Try different optimizers
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)
    lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, factor=0.9, patience=40, cooldown=40)

    pbar = tqdm(range(epochs))
    for ep in pbar:
        # print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        # newLR = lr*(1-ep/epochs)

        newLR = optim.param_groups[0]['lr']

        # pbar = tqdm(dataloader)
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            optim.zero_grad()

            loss = ddpm(features, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1)

            optim.step()
            pbar.set_description(f"loss: {loss.item():.4f}, lr: {newLR:.6f}")

        lrScheduler.step(loss.item())

        if ep%80==0 or ep == int(epochs-1):
            ddpm.eval()
            with torch.no_grad():
                numRows = 4
                n_sample = numRows*numClasses
                for w_i, w in enumerate(guidanceStrengths):
                    x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), classifierGuidance=w)

                    # append some real images at bottom, order by class also
                    x_real = torch.Tensor(x_gen.shape).to(device)
                    for k in range(numClasses):
                        for j in range(int(n_sample/numClasses)):
                            try: 
                                idx = torch.squeeze((labels == k).nonzero())[j]
                            except:
                                idx = 0
                            x_real[k+(j*numClasses)] = features[idx]

                    x_all = torch.cat([x_gen, x_real])
                    grid = make_grid(x_all*-1 + 1, nrow=numClasses)
                    save_image(grid, savePath + f"image_ep{ep}_w{w}.png")
                    print('saved image at ' + savePath + f"image_ep{ep}_w{w}.png")

                    # if ep%40==0 or ep == int(epochs-1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample/numClasses), ncols=numClasses,sharex=True,sharey=True,figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/numClasses)):
                            for col in range(numClasses):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*numClasses)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    ani.save(savePath + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + savePath + f"gif_ep{ep}_w{w}.gif")
                    print()
                    
        # optionally save model
        if save_model and ep == int(epochs-1):
            torch.save(ddpm.state_dict(), savePath + f"model_{ep}.pth")
            print('saved model at ' + savePath + f"model_{ep}.pth")
    





def main():
    
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    BATCH_SIZE = 16
    INPUT_SIZE = (BATCH_SIZE, 1, 28, 28)
    NUM_CLASSES = 10

    randomLabels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)).to(device)
    randomTimes = torch.randint(0, 100, (BATCH_SIZE,)).to(device).to(torch.float32)
    classMasks = torch.ones((BATCH_SIZE,)).to(device)

    dummyInput = torch.rand(INPUT_SIZE).to(device)

    module = UNet(numClasses=NUM_CLASSES)

    # profileModel(module, input_size=INPUT_SIZE)

    # Create an instance of the nn.module class defined above:
    module = module.to(device)

    output = module.forward(dummyInput, randomLabels, randomTimes, classMasks)
    if output is not None:
        print(output.shape)
    
    
    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    # dataset = trainset
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    trainDDPM(10, 20, 256, 400, dataset)

    
    # # return
    
    # # hardcoding these here
    # n_epoch = 20
    # batch_size = 128*3
    # numTimesteps = 400 # 500
    # device = "cuda:0"
    # lr = 1e-4
    # save_model = False
    # savePath = './DiffusionData/'
    # guidanceStrengths = [0.0, 0.5, 2.0] # strength of generative guidance

    # ddpm = DDPM(model=UNet(numClasses=NUM_CLASSES), betas=(1e-4, 0.02), numTimesteps=numTimesteps, dropoutRate=0.4, device=device)
    # ddpm.to(device)

    # # optionally load a model
    # # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    # tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    # dataset = MNIST("./data", train=True, download=True, transform=tf)
    # # dataset = trainset
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    # for ep in range(n_epoch):
    #     print(f'epoch {ep}')
    #     ddpm.train()

    #     # linear lrate decay
    #     optim.param_groups[0]['lr'] = lr*(1-ep/n_epoch)

    #     pbar = tqdm(dataloader)
    #     for x, c in pbar:
    #         optim.zero_grad()
    #         x = x.to(device)
    #         c = c.to(device)
    #         loss = ddpm(x, c)
    #         loss.backward()

    #         pbar.set_description(f"loss: {loss.item():.4f}")
    #         optim.step()
    
    #     ddpm.eval()
    #     with torch.no_grad():
    #         n_sample = 4*NUM_CLASSES
    #         for w_i, w in enumerate(guidanceStrengths):
    #             x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), classifierGuidance=w)

    #             # append some real images at bottom, order by class also
    #             x_real = torch.Tensor(x_gen.shape).to(device)
    #             for k in range(NUM_CLASSES):
    #                 for j in range(int(n_sample/NUM_CLASSES)):
    #                     try: 
    #                         idx = torch.squeeze((c == k).nonzero())[j]
    #                     except:
    #                         idx = 0
    #                     x_real[k+(j*NUM_CLASSES)] = x[idx]

    #             x_all = torch.cat([x_gen, x_real])
    #             grid = make_grid(x_all*-1 + 1, nrow=10)
    #             save_image(grid, savePath + f"image_ep{ep}_w{w}.png")
    #             print('saved image at ' + savePath + f"image_ep{ep}_w{w}.png")

    #             if ep%5==0 or ep == int(n_epoch-1):
    #                 # create gif of images evolving over time, based on x_gen_store
    #                 fig, axs = plt.subplots(nrows=int(n_sample/NUM_CLASSES), ncols=NUM_CLASSES,sharex=True,sharey=True,figsize=(8,3))
    #                 def animate_diff(i, x_gen_store):
    #                     print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
    #                     plots = []
    #                     for row in range(int(n_sample/NUM_CLASSES)):
    #                         for col in range(NUM_CLASSES):
    #                             axs[row, col].clear()
    #                             axs[row, col].set_xticks([])
    #                             axs[row, col].set_yticks([])
    #                             # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
    #                             plots.append(axs[row, col].imshow(-x_gen_store[i,(row*NUM_CLASSES)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
    #                     return plots
    #                 ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
    #                 ani.save(savePath + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
    #                 print('saved image at ' + savePath + f"gif_ep{ep}_w{w}.gif")
    #     # optionally save model
    #     if save_model and ep == int(n_epoch-1):
    #         torch.save(ddpm.state_dict(), savePath + f"model_{ep}.pth")
    #         print('saved model at ' + savePath + f"model_{ep}.pth")
















if __name__ == '__main__':
    main()