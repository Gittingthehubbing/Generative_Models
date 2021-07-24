# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:06:49 2021

@author: Quiet

Tries to reproduce and improve the tandem network in 
Training Deep Neural Networks for the Inverse Design of Nanophotonic Structures
liu_training_2018 (https://doi.org/10.1021/acsphotonics.7b01377)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as t
import torchvision.transforms as tr
from torch.utils.data.dataloader import DataLoader as dl
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data  import random_split #for train test split
import torch.nn as nn
from tqdm import tqdm
from scipy.ndimage.filters import uniform_filter1d #for numpy rolling average
from scipy import signal #to make gaussian
from scipy import constants
from sklearn.preprocessing import StandardScaler as sScaler
import optuna
from optuna.trial import TrialState
import os
from torch.utils.tensorboard import SummaryWriter
import datetime


def createFolder(ResultsFolderName):
    if not os.path.exists(ResultsFolderName):
        os.mkdir(ResultsFolderName)
        print(ResultsFolderName+ ' Folder Created')
    else:
        print(ResultsFolderName+'Folder exists')
        
        
def define_model_Best(pDict):
    n_layers = pDict["nL"]    
    layers = []
    in_features = dShape
    i=0
    for i in range(1,n_layers):
        out_features = pDict["n{}".format(i)]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.Dropout(pDict[f"p{i}"]))
        layers.append(getattr(nn, pDict[f"a{i}"])())
        in_features = out_features
    layers.append(nn.Linear(in_features, rShape))
    return nn.Sequential(*layers)


class dToR(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()
        self.xSize = x.shape[1]
        try:
            self.ySize = y.shape[1]
        except:
            self.ySize = 1
        print('Classi x Shape',self.xSize,'yShape',self.ySize)  
        self.fc1 = nn.Linear(self.xSize,50)
        self.fc2 = nn.Linear(50,20)
        self.fc3 = nn.Linear(20,20)
        self.fc4 = nn.Linear(20,20)
        self.fc5 = nn.Linear(20,self.ySize)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        #x = self.sigmoid(x)
        #x = self.relu(x)
        return x
    
class dToRSeq(nn.Module):
    
    
    def __init__(self,x,y):
        super().__init__()
        self.xSize = x.shape[1]
        try:
            self.ySize = y.shape[1]
        except:
            self.ySize = 1
        print('Classi x Shape',self.xSize,'yShape',self.ySize)  
        self.model = nn.Sequential(
            nn.Linear(self.xSize,500),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.BatchNorm1d(500),
            nn.Linear(500,200),
            nn.ReLU(),            
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200,self.ySize))
    
    def forward(self,x):
        x = self.model(x)
        return x
    
class rToD(nn.Module):
    
    
    def __init__(self,x,y):
        super().__init__()
        self.leakyRelu = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        #random numbers in, flat image out
        self.xSize = x.shape[1]
        self.ySize = y.shape[1]
        print('Gen x Shape',self.xSize,'yShape',self.ySize)
        self.fc1 = nn.Linear(self.xSize,200)
        self.fc2 = nn.Linear(200,500)
        self.fc3 = nn.Linear(500,200)
        self.fc4 = nn.Linear(200,self.ySize)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        return x
    
def makeDSet(df,xName, yName, mean=0.5, std=0.25):
    x = df[xName].to_numpy()
    x = np.stack(x)
    x /= np.amax(x)    
    x = t.tensor(x).float() 
    y = df[yName].to_numpy()
    y = np.stack(y)
    y /= np.amax(y)
    y = t.tensor(y).float()
    dSet = TensorDataset(x,y)
    return dSet
        
def makeGridTest(model,loader,name,gridSize = 3,savefig = False):
    model.eval()
    fig, axs = plt.subplots(nrows=gridSize,ncols=gridSize,sharex='col',sharey='row',dpi=200)
    num = 0
    for ax, (xT2, yT2) in zip(axs.ravel(),loader):
        with t.no_grad():
            testOutSpecDtoR = model.forward(xT2.to(device)).cpu().detach().numpy()
            lossSingle = criticDToR(t.tensor(testOutSpecDtoR[:1]),yT[:1].cpu().detach())
        #ax = fig.add_subplot(4,4,1+iT)
        ax.plot(wvls,yT2[0],label='Target ')
        ax.plot(wvls,testOutSpecDtoR[0],label='Fake Spectrum')   
        ax.set_title(f'Loss: {lossSingle:.2e}')
        if num == gridSize**2-1:
            ax.legend()
            break    
        num+=1
    if savefig:
        plt.savefig(f'{folderName}/ForwardNetGrid_{name}.png',dpi=300)
        plt.close()
    return fig

infile = 'ResDfTransmissionTMM_5LayerPairsNormalDistr_50kSamples.pkl'
wvlFile = 'wvls.csv'

folderName = 'Output/Torch_Train_LeakyRelu100Epochs_'+infile[:infile.find('.pkl')]

retrainDToR = True
retrainRToD = True
testSingleOverfit = False

propertiesDict={
    "nSamples":50000,
    "batchSize":2**10,
    "testSplit":0.35,
    "epochs":200,
    "opti":"Adam",
    "lossF":"MSE",
    "lr":5.9194e-4,
    "weightDecay":1e-4,
    "nL":5,
    "inShape":2,
    "outShape":50,
    "n1":2**10,
    "a1":'LeakyReLU',
    "p1":0.2,
    "n2":2**10,
    "a2":'LeakyReLU',
    "p2":0.2,
    "n3":2**10,
    "a3":'LeakyReLU',
    "p3":0.3,
    "n4":2**11,
    "a4":'LeakyReLU',
    "p4":0.5,}


createFolder(folderName)
inDf = pd.read_pickle(f"training_data/{infile}").iloc[:propertiesDict["nSamples"]]
numSamples = len(inDf)

dSet = makeDSet(inDf, "Thicknesses", "TransTE")

trainSplit = 1- propertiesDict["testSplit"]
numTrSamples = int(trainSplit*numSamples)
trainSet, testSet = random_split(dSet,[numTrSamples,numSamples-numTrSamples])

trainLoader = dl(trainSet,batch_size=propertiesDict["batchSize"])
testLoader = dl(testSet,batch_size=propertiesDict["batchSize"])

firstX, firstY = next(iter(trainLoader))

dShape = firstX.shape[1]
wvls = pd.read_csv(f"training_data/{wvlFile}").loc[:,"Wvl"]
rShape = wvls.shape[0]


device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    

log_dir = "logs/fit_PyTorch/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"lr_e-4_Batch1024_20kSamples"
log_dirRToD = "logs/fit_PyTorch/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"lr_e-4_Batch1024_20kSamplesRToD"

writer = SummaryWriter(log_dir)
writerRToD = SummaryWriter(log_dirRToD)



if retrainDToR:
    #dToRNet = dToRSeq(t.tensor(firstX).float(),t.tensor(firstY).float()).to(device)
    dToRNet = define_model_Best(propertiesDict).to(device)
else:
    dToRNet = t.load('dToRNet.pkl')

# writer.add_graph(dToRNet,firstX.to(device))
# writer.close()

criticDToR = nn.MSELoss()
optiDToR = t.optim.Adam(dToRNet.parameters(),lr=propertiesDict["lr"],weight_decay=propertiesDict["weightDecay"])
epochs = propertiesDict["epochs"]
batchSize = propertiesDict["batchSize"]

lossDf = pd.DataFrame()
evalDf = pd.DataFrame()

epochLossDfTrainDToR = []
epochLossDfEvalDToR = []

if testSingleOverfit:
    dToRNet.train()
    epochs = 500
    
    num=0
    for e in tqdm(range(epochs)):
        x, y = next(iter(trainLoader))
        x, y = x[:2], y[:2]
        x[1], y[1] = x[0], y[0]
        num+=1
        optiDToR.zero_grad()
        out = dToRNet.forward(x.to(device))
        loss = criticDToR(out,y.to(device))
        lossSqrt = loss
        lossSqrt.backward()
        optiDToR.step()
        tempDf = pd.DataFrame({'Loss':lossSqrt.cpu().detach().numpy()},index=[num])
        lossDf = lossDf.append(tempDf)
        
    
    generated = dToRNet.forward(x.to(device))
    forPlotFake = generated.cpu().detach().numpy()[0]
    forPlotReal = y[0].cpu().detach().numpy()
    lossForPlot = criticDToR(generated.detach()[0],y[0].to(device).detach())
    fig,axs = plt.subplots(2,1,figsize=(6,8),dpi=300)
    axs[0].plot(lossDf.rolling(10).mean().values,'k-',label='TrainLoss')
    axs[0].legend()
    axs[0].set_yscale('log')
    axs[1].plot(wvls,forPlotFake,'k.',label='fake')
    axs[1].plot(wvls, forPlotReal,'r-',label='real')
    axs[1].legend()
    plt.title(f'Forward Network with Loss {lossForPlot:.3e}')
    plt.show()
    plt.close()
else:
    if retrainDToR:
        dToRNet.train()
        for e in tqdm(range(epochs)):
            trainLossTempDToR = pd.DataFrame()
            evalLossTempDToR = pd.DataFrame()
            for i, (x, y) in enumerate(trainLoader):   
                optiDToR.zero_grad()
                out = dToRNet.forward(x.to(device))
                loss = criticDToR(out,y.to(device))
                lossSqrt = loss
                lossSqrt.backward()
                optiDToR.step()
                tempDf = pd.DataFrame({'Loss':lossSqrt.cpu().detach().numpy()},index=[i])
                lossDf = lossDf.append(tempDf)
                trainLossTempDToR = trainLossTempDToR.append(tempDf)
                if i%(int(len(trainLoader)/2)) == 0:
                    with t.no_grad():
                        dToRNet.eval()
                        for iT, (xT, yT) in enumerate(testLoader):
                            outTest = dToRNet.forward(xT.to(device))
                            lossTest = criticDToR(outTest,yT.to(device))
                            #lossTest = lossTest
                            tempDfT = pd.DataFrame({'lossTest':lossTest.cpu().detach().numpy()},index=[i+iT])
                            evalDf = evalDf.append(tempDfT)
                            evalLossTempDToR = evalLossTempDToR.append(tempDfT)
                    dToRNet.train()
            epochLossDfTrainDToR.append(trainLossTempDToR.mean().values.item())
            epochLossDfEvalDToR.append(evalLossTempDToR.mean().values.item())
            writer.add_scalar("TrainLossEpoch", trainLossTempDToR.mean().values.item(),e)
            writer.add_scalar("EvalLossEpoch", evalLossTempDToR.mean().values.item(),e)
            if e%50 == 0:
                writer.add_figure('Test',makeGridTest(dToRNet, testLoader,name = "Test_Data"),global_step=e)    
                plt.close()
        
        
        t.save(dToRNet,log_dir+'/dToRNet.pkl')
        dToRNet.eval()
        makeGridTest(dToRNet,testLoader,"Test_Data",3,True)
        makeGridTest(dToRNet,trainLoader,"Train_Data",3,True)
        
        dNoAir = xT.to(device)[:2]
        with t.no_grad():
            generated = dToRNet.forward(dNoAir)
            forPlotFake = generated.cpu().detach().numpy()[0]
            forPlotFake = uniform_filter1d(forPlotFake, size=5)  
            forPlotReal = yT[0].cpu().detach().numpy()
            #plotCritic = criticDToR(forPlotReal,forPlotFake)
        
        lossforplot = np.sqrt(np.mean(np.subtract(forPlotReal,forPlotFake)**2)/2)
        
        fig,axs = plt.subplots(2,1,figsize=(6,8),dpi=300)
        # axs[0].plot(lossDf.rolling(10).mean().values,'k-',label='TrainLoss')
        # axs[0].plot(evalDf.rolling(10).mean().values,'r-',label='TestLoss')
        axs[0].plot(epochLossDfTrainDToR,'k.-',label='TrainLoss')
        axs[0].plot(epochLossDfEvalDToR,'r.-',label='TestLoss')
        axs[0].legend()    
        axs[0].set_yscale('log')
        axs[0].set_ylim((epochLossDfTrainDToR[-1],epochLossDfTrainDToR[5]))
        axs[1].plot(wvls,forPlotFake,label=f'fake with Loss {lossforplot:.3f}')
        axs[1].plot(wvls, forPlotReal)
        axs[1].legend()
        plt.title('Forward Network')
        plt.savefig(f'{folderName}/ForwardNet.png',dpi=300)
        plt.show()
        plt.close()

    if retrainRToD:
        gaussSpec = (1 - signal.gaussian(rShape,std=10)*0.8)
        gaussSpecArr = np.stack([gaussSpec,gaussSpec])
        
        rToDNet = rToD(firstY, firstX).to(device)
        criticRToD = nn.MSELoss()
        optiRToD = t.optim.Adam(rToDNet.parameters(),lr=3e-4)
        
        lossDf2 = pd.DataFrame()
        evalDf2 = pd.DataFrame()
        epochLossDfTrain = []
        epochLossDfEval = []
        dToRNet.eval() # makes sure dropout layers are not used during prediction
        for e in tqdm(range(epochs)):
            trainLossTemp = pd.DataFrame()
            evalLossTemp = pd.DataFrame()
            for i2, (x2, y2) in enumerate(trainLoader):
                rToDNet.train()
                optiRToD.zero_grad()
                out2 = rToDNet.forward(y2.to(device))
                dToRout = dToRNet.forward(out2)
                loss2 = criticRToD(dToRout,y2.to(device))
                #loss2 = loss2*len(x2)
                loss2.backward()
                optiRToD.step()
                tempDf = pd.DataFrame({'Loss':loss2.cpu().detach().numpy()},index=[i2])
                lossDf2 = lossDf2.append(tempDf)
                trainLossTemp = trainLossTemp.append(tempDf)
                if i2%(len(x2)*2) == 0:
                    with t.no_grad():
                        for iT, (xT2, yT2) in enumerate(testLoader):
                            rToDNet.eval()
                            outTest = rToDNet.forward(yT2.to(device))
                            dToRoutTest = dToRNet.forward(outTest)
                            lossTest2 = criticRToD(dToRoutTest,yT2.to(device))
                            #lossTest2 = lossTest*len(xT2)
                            tempDfT = pd.DataFrame({'lossTest':lossTest2.cpu().detach().numpy()},index=[i2+iT])
                            evalDf2 = evalDf2.append(tempDfT)
                            evalLossTemp = evalLossTemp.append(tempDfT)
            epochLossDfTrain.append(trainLossTemp.mean().values.item())
            epochLossDfEval.append(evalLossTemp.mean().values.item())
            writerRToD.add_scalar("TrainLossEpoch", trainLossTemp.mean().values.item(),e)
            writerRToD.add_scalar("EvalLossEpoch", evalLossTemp.mean().values.item(),e)
            #print(e,' lossTest',lossTest)
        
        # plt.plot(lossDf2.rolling(10).mean().values,'k-',label='TrainLossTandem')
        # plt.plot(evalDf2.rolling(10).mean().values,'r-',label='TestLossTandem') 
        plt.plot(epochLossDfTrain,'k.-',label='TrainLossTandem')
        plt.plot(epochLossDfEval,'r.-',label='TestLossTandem')    
        plt.yscale('log')
        plt.legend()
        plt.title('Tandem Network')
        
        plt.savefig(f'{folderName}/TandemNet.png',dpi=300)
        #plt.show()
        plt.close()
        
        hardTest = np.ones_like(firstY[:1])
        hardTest[:,150:200] = 0
        #hardTest = np.expand_dims(hardTest,1)
        rToDNet.eval()
        with t.no_grad():
            hardThick = rToDNet.forward(t.tensor(hardTest).float().to(device))
            hardSpecPred = dToRNet.forward(hardThick).cpu().detach().numpy()
            
        plt.plot(wvls,hardTest[0],label='Real')
        plt.plot(wvls,hardSpecPred[0],label='Fake')
        plt.legend()
        #plt.show()
        plt.close()
        
        with t.no_grad():
            gaussOutThick = rToDNet.forward(t.tensor(gaussSpecArr).float().to(device))
            gaussOut = dToRNet.forward(gaussOutThick).cpu().detach().numpy()
        
        plt.plot(wvls,gaussOut[0])
        plt.plot(wvls,gaussSpec)
        #plt.show()
        plt.close()
        
        gridSize = 3
        fig, axs = plt.subplots(nrows=gridSize,ncols=gridSize,sharex='col',sharey='row')
        # fig = plt.figure()
        num = 0
        for ax, (xT2, yT2) in zip(axs.ravel(),testLoader):
            with t.no_grad():
                testOutThick = rToDNet.forward(yT2.to(device))
                testOutSpecRtoD = dToRNet.forward(testOutThick).cpu().detach().numpy()
                testOutSpecDtoR = dToRNet.forward(xT2.to(device)).cpu().detach().numpy()
            #ax = fig.add_subplot(4,4,1+iT)
            ax.plot(wvls,yT2[0],label='Target Spectrum')
            ax.plot(wvls,testOutSpecDtoR[0],label='Spectrum From\n Real Thicknesses')       
            ax.plot(wvls,testOutSpecRtoD[0],label='Spectrum From\n Generated Thicknesses') 
            if num == gridSize**2-1:
                ax.legend()
                break    
            num+=1
        
        plt.savefig(f'{folderName}/TandemGrid.png',dpi=300)
        plt.show()
        plt.close()
        