import matplotlib.pyplot as plt 
EPOCHS = 16
LR = 1e-2


import numpy as np
import torch
def train(model, loader, epochs = EPOCHS, lr=LR):

    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        losses = []

        for i, (d, t) in enumerate(loader):
            optim.zero_grad()

            v, gibbs = model(d.view(-1, 784)) #flatten the image

            loss = model.free_energy(v) - model.free_energy(gibbs)
            losses.append(loss.item())
            loss.backward()
            optim.step()

        
        print(f'EPOCH: {epoch+1} | LOSS: {np.mean(losses)}')
    
    return model

def save(image, fname, save = True):

    image = np.transpose(image.numpy(), (1, 2, 0))
    
    plt.imsave(fname, image)
