import torch as t
import torch.nn as nn
import torchvision
import torchvision.transforms as tr
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader as dl
import pathlib as pl
from PIL import Image
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class vae(nn.Module):

    def __init__(self, images, latentDim):

        super().__init__()

        self.encModel = nn.Sequential(            
            self.block(images.shape[1],128,4,2,1),
            self.block(128,256,4,2,1),
            self.block(256,256,4,2,1),
            self.block(256,128,4,2,1),
            self.block(128,64,4,2,1),
            self.block(64,64,4,2,1))
        #,nn.Conv2d(64, 1, 6, 1, 1)

        #self.batchSize = images.shape[0]
        #self.flatSize = images.view(self.batchSize,-1).shape[1]
        self.latentDim = latentDim

        #self.fc11 = nn.Linear(self.flatSize,int(self.flatSize/2))
        #self.fc12 = nn.Linear(int(self.flatSize/2),int(self.latentDim*2))
        self.fc12 = nn.Linear(1024,int(self.latentDim*2))

        
        self.decModel = nn.Sequential(
            self.genBlock(self.latentDim,256,4,1,0),
            self.genBlock(256,256,4,2,1),
            self.genBlock(256,256,4,2,1),
            self.genBlock(256,256,4,2,1),
            self.genBlock(256,128,4,2,1),
            self.genBlock(128,64*2,4,2,1),
            nn.ConvTranspose2d(64*2,images.shape[1],4,2,1),
            nn.Sigmoid())


        #self.fc21 = nn.Linear(int(self.latentDim),int(self.flatSize/2))
        #self.fc22 = nn.Linear(int(self.flatSize/2),self.flatSize)


    def genBlock(self, inchannel, outchannel, kernel, stride, padding):
        block = nn.Sequential(
            nn.ConvTranspose2d(inchannel, outchannel, kernel, stride, padding),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(0.2))
        return block

    def block(self, inchannel, outchannel, kernel, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel, stride, padding,bias=False),
            nn.InstanceNorm2d(outchannel, affine=True),
            nn.LeakyReLU(0.2))
        return block

    def reparam(self, mu, logVar):

        stanDev = t.exp(1/2 * logVar)
        eps = t.randn_like(stanDev)
        return mu + eps * stanDev

    def encode(self,x):

        x = nn.ReLU()(self.encModel(x))
        x = nn.Flatten(1)(x)
        x = self.fc12(x).view(-1, 2, self.latentDim)

        mu = x[:,0,:]
        logVar = x[:,1,:]

        latentVec = self.reparam(mu, logVar).view(-1, self.latentDim,1,1)
        #latentVec = t.unsqueeze(latentVec,2)
        #t.randn(xBatch.shape[0],latentDim,1,1)

        return mu, logVar, latentVec

    def decode(self, latentVec):

        #x = nn.ReLU()(self.fc21(latentVec))
        #nn.Sigmoid()(self.fc22(x))
        return self.decModel(latentVec)

class dSet(Dataset):

    def __init__(self, imPaths, transforms = None,simpleTransform=False):
        super().__init__()
        self.imPaths = imPaths
        self.transforms = transforms
        self.simpleTransform = simpleTransform

    def __len__(self):
        return len(self.imPaths)

    def __getitem__(self, index):
        im = Image.open(self.imPaths[index]).convert("RGB")
        if self.transforms is not None:
            im = self.transforms(im)
        if self.simpleTransform:  
            im = tr.Compose([
                tr.Resize(imSize),tr.ToTensor()])(im)
        return im
        


inImDir = r"F:\TEMPPP\varFDTD2"
outImDir = "outIms"

stringForSearch = "FORCUDA"
imExtension = "png"

imSize = 256
latentDimension = 2**4
learningRate = 6e-4
gammaSched = 1-5e-5
batchSize = 2**3
epochs = 2
checkNorm = 0

imPaths = [f for f in pl.Path(inImDir).glob(f"*/*{stringForSearch}*.{imExtension}")]

transformsForImages = tr.Compose([
    tr.Resize(imSize),tr.ToTensor(),
    tr.Normalize((
            0.45709094405174255,
            0.45657968521118164,
            0.45707881450653076),
            (0.28522488474845886,
              0.28490182757377625,
              0.2852752208709717)),])
tensorSet = dSet(imPaths,None,True)
dLoader = dl(tensorSet,batchSize,True)
initBatch = next(iter(dLoader))


if checkNorm:
    for axI in range(3):
        meanList=[]
        stdList=[]
        for xS in tensorSet:
            meanList.append(tensorSet[axI].mean().cpu().detach().numpy().item())
            stdList.append(tensorSet[axI].std().cpu().detach().numpy().item())

        print(axI,'Mean ',np.mean(meanList))
        print(axI,'Std ',np.mean(stdList)) #used to compute normalisation when normalisation is turned off


d = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

vaeNet = vae(initBatch, latentDimension).to(d)
opti = t.optim.Adam(vaeNet.parameters(),learningRate)
scheduler = t.optim.lr_scheduler.ExponentialLR(opti,gammaSched)
criterion = nn.BCELoss(reduction="sum")

trLosses = []
criterionLosses = []
for e in range(epochs):
    for imBatch in dLoader:
        vaeNet.train()
        # xFlat = imBatch.view(imBatch.shape[0],-1).to(d)
        xBatch = imBatch.to(d)
        mu, logVar, latentVec = vaeNet.encode(xBatch)
        out = vaeNet.decode(latentVec)
        trainLoss = criterion(out, xBatch)
        criterionLosses.append(trainLoss.cpu().detach().numpy().item())
        trainLoss += -1/2 * t.sum(1+logVar - mu**2 - logVar.exp())
        opti.zero_grad()
        trainLoss.backward()
        opti.step()
        scheduler.step()
        trLosses.append(trainLoss.cpu().detach().numpy().item())
    print("Trainloss ",trainLoss.cpu().detach().numpy().item())

criterionLossesDf = pd.DataFrame(criterionLosses)
trLossDf = pd.concat([pd.DataFrame(trLosses),criterionLossesDf],axis=1)
trLossDf.columns=["FullLoss", "CriterionLoss"]
trLossDf.plot()
#plt.yscale("log")
plt.show()

vaeNet.eval()
with t.no_grad():
    latentVecTest = t.randn(16, latentDimension,1,1).to(d)
    outTest = vaeNet.decode(latentVecTest)
img_grid_fakeTest = torchvision.utils.make_grid(
    outTest, normalize=True)
gridFIm = tr.ToPILImage()(img_grid_fakeTest)
gridFIm.show()

img_grid_real = torchvision.utils.make_grid(
    xBatch.to(d)[:16], normalize=True)
img_grid_fake = torchvision.utils.make_grid(
    out[:16], normalize=True)
gridFIm = tr.ToPILImage()(img_grid_fake)
gridFIm.show()

gridRIm = tr.ToPILImage()(img_grid_real)
gridRIm.show()

# lastImOut = tr.ToPILImage()(out[0].cpu().detach().view(-1,imSize,imSize))
# lastImOut.show()
print("all done")