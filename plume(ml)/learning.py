#!/usr/bin/env python
# coding: utf-8

import os, sys, random
import numpy as np
import matplotlib.pyplot as plt
import pylab
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd 
import torch.utils.data 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torchviz import make_dot



#plt.style.use("dark_background")
dpi = 100
def ani_frame(X,N_TIME_STEPS,name,dataset,norm=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(X[5,:,:],origin='lower',interpolation='bicubic',cmap='jet')
    fig.colorbar(im)
    
    # if name == 'T' or name == 'Tnd':
    #     im.set_clim([0,4e3])
    # elif name == 'radVar1' or name == 'radVar1nd' or name == 'radVar1test':
    #     im.set_clim([0,3e5])
    # elif name == 'no' or name == 'nond':
    #     im.set_clim([0,0.6])
    # elif name == 'radVar2' or name == 'radVar2nd' or name == 'radVar2test':
    #     im.set_clim([0,8e5])
    # elif name == 'DelQ' or name == 'DelQnd':
    #     im.set_clim([0,8e5])
    # elif norm==True:
    #     im.set_clim([0,1])
    # else:
    #     im.set_clim([0,2.5])
    test = X[X!=0]
    print('mean/stddev', np.mean(test), np.std(test))
    
    im.set_clim([0,np.amax(X)])
    
    fig.set_size_inches([5,5])
    pylab.tight_layout()

    def update_img(n):
        tmp = X[n,:,:]
        im.set_data(tmp)
        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,N_TIME_STEPS,interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save(dataset+name+'_demo.mp4',writer=writer,dpi=dpi)
    return ani

def getani(no,T,radVar1,radVar2,Delq,kPlanck,ns,dataset,which,norm=False):
    if which == 'nd':
        ani_frame(T,ns,'Tnd',dataset,norm)
        ani_frame(no,ns,'nond',dataset,norm)
        ani_frame(radVar1,ns,'radVar1nd',dataset,norm)
        ani_frame(radVar2,ns,'radVar2nd',dataset,norm)
        ani_frame(Delq,ns,'DelQnd',dataset,norm)
        ani_frame(kPlanck,ns,'kPlancknd',dataset,norm)
    elif which == 'testresults':
        ani_frame(radVar1,ns,'radVar1test',dataset,norm)
        ani_frame(radVar2,ns,'radVar2test',dataset,norm)
    elif which == 'errorAbs':
        ani_frame(radVar1,ns,'radVar1ErrorAbs',dataset,norm)
        ani_frame(radVar2,ns,'radVar2ErrorAbs',dataset,norm)
    else:
        ani_frame(T,ns,'T',dataset,norm)
        ani_frame(no,ns,'no',dataset,norm)
        ani_frame(radVar1,ns,'radVar1',dataset,norm)
        ani_frame(radVar2,ns,'radVar2',dataset,norm)
        ani_frame(Delq,ns,'DelQ',dataset,norm)
        ani_frame(kPlanck,ns,'kPlanck',dataset,norm)
    return


def loaddata(dataset,var):
    varz = np.load(dataset+var+'.npz')
    var = np.transpose(varz['arr_0'], (2,0,1))
    return var

def getdata(dataset):
    no = loaddata(dataset,'no')
    T = loaddata(dataset,'T')
    radVar1 = loaddata(dataset,'radVar1')
    radVar2 = loaddata(dataset,'radVar2')
    Delq = loaddata(dataset,'DelQ')
    kPlanck = loaddata(dataset,'kPlanck')
    return no, T, radVar1, radVar2, Delq, kPlanck

def getnorm(no,T,radVar1,radVar2,Delq,kPlanck,which):
    if which == 'target':
        radVar1 = normalize(radVar1,which)
        radVar2 = normalize(radVar2,which)
        return radVar1,radVar2
    elif which == 'input':
        no = normalize(no,which)
        T = normalize(T,which)
        Delq = normalize(Delq,which)
        kPlanck = normalize(kPlanck,which)
        return no,T,Delq,kPlanck
    else:
        no = normalize(no,which)
        T = normalize(T,which)
        radVar1 = normalize(radVar1,which)
        radVar2 = normalize(radVar2,which)
        Delq = normalize(Delq,which)
        kPlanck = normalize(kPlanck,which)
        return no,T,radVar1,radVar2,Delq,kPlanck

def transform(A,which):
    
    Amax = np.amax(A)
    Amin = np.amin(A)
    Range = Amax - Amin
    
    if which == 'ilal':
        #  inverse logit after log
        var = A / (1 + A)
    elif which == 'silal':
        # square inverse logit after log
        var = A*A / (1 + A*A)
    elif which == 'exp':
        # exponential 
        var = 1 - np.exp(A)
    elif which == 'norm':
        # norm 0 to 1
        var = (A - Amin)/Range
    
    return var

def detransform(A,zA,which):
    
    Amax = np.amax(zA);
    Amin = np.amin(zA)
    Range = Amax - Amin
    print('mix/max original', Amin, Amax)
    print('min/max transformed',np.amin(A),np.amax(A))

    if which == 'ilal':
        # inverse logit after log
        var = A*(1 + zA)
    elif which == 'silal':
        # square inverse logit after log
        var = np.sqrt(A*A*(1 + zA*zA))
    elif which == 'exp':
        # exponential 
        var = A + np.exp(zA) - 1
    elif which == 'norm':
        # norm 0 to 1
        var = A*Range + Amin
        
    print('min/max detransformed',np.amin(var),np.amax(var))
    return var 

def normalize(A,which):
    if which == 'target':
        Anrm = transform(A,'ilal')
    else:
        Anrm = transform(A,'norm')
    return Anrm

def denormalize(A,zA,which):
    if which == 'target':
        Adnrm = detransform(A,zA,'ilal')
    else:
        Adnrm = detransform(A,zA,'norm')
    return Adnrm


def getdenorm(zno,zT,zradVar1,zradVar2,zDelq,zkPlanck,no,T,radVar1,radVar2,Delq,kPlanck,which):
    if which == 'target':
        radVar1 = denormalize(radVar1,zradVar1,which)
        radVar2 = denormalize(radVar2,zradVar2,which)
        return radVar1,radVar2
    elif which == 'input':
        no = denormalize(no,zno,which)
        T = denormalize(T,zT,which)
        Delq = denormalize(Delq,zDelq,which)
        kPlanck = denormalize(kPlanck,zkPlanck,which)
        return no,T,Delq,kPlanck
    else:
        no = denormalize(no,zno,which)
        T = denormalize(T,zT,which)
        radVar1 = denormalize(radVar1,zradVar1,which)
        radVar2 = denormalize(radVar2,zradVar2,which)
        Delq = denormalize(Delq,zDelq,which)
        kPlanck = denormalize(kPlanck,zkPlanck,which)
        return no,T,radVar1,radVar2,Delq,kPlanck

def znorm(A):
    Amax = np.amax(A)
    Amin = np.amin(A)
    Range = Amax - Amin
    Anrm = (A - Amin)/Range
    return Anrm


# helper to show target channels: normalized, with colormap, side by side
def showSbs(inputs,targets,i):
  f, ax = plt.subplots(2,2)
  f.tight_layout()
  [axi.set_axis_off() for axi in ax.ravel()]
  
  im = ax[0,0].imshow(inputs[i,0,:,:],origin='lower',cmap='jet')
  im.set_clim([0,np.amax(inputs[i,0,:,:])])
  im = ax[0,1].imshow(inputs[i,1,:,:],origin='lower',cmap='jet')
  im.set_clim([0,np.amax(inputs[i,1,:,:])])
  im = ax[1,0].imshow(targets[i,0,:,:],origin='lower',cmap='jet')
  im.set_clim([0,np.amax(targets[i,0,:,:])])
  im = ax[1,1].imshow(targets[i,1,:,:],origin='lower',cmap='jet')
  im.set_clim([0,np.amax(targets[i,1,:,:])])

  f.subplots_adjust(right=0.8)
  cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
  f.colorbar(im, cax=cbar_ax)


#dataset = 'd300s50/'
#dataset = 'd300s50na/'
dataset = 'd300s200/'

no,T,radVar1,radVar2,Delq,kPlanck = getdata(dataset+'train/')
zno,zT,zradVar1,zradVar2,zDelq,zkPlanck = no,T,radVar1,radVar2,Delq,kPlanck

ns = len(no[:,0,0])
nps = len(no[0,:,0])

print("number of time steps, ns: ",ns)
print("number pixels, nps: ",nps)
print("max no", np.amax(no))
print("max T", np.amax(T))
print("max radVar1", np.amax(radVar1))
print("max radVar2", np.amax(radVar2))
# print("max Delq", np.amax(Delq))
# print("max kPlanck", np.amax(kPlanck))

getani(no,T,radVar1,radVar2,Delq,kPlanck,ns,dataset,'s',norm=False)

no,T,Delq,kPlanck = getnorm(no,T,radVar1,radVar2,Delq,kPlanck,'input')
radVar1,radVar2 = getnorm(no,T,radVar1,radVar2,Delq,kPlanck,'target')


print("max no", np.amax(no))
print("max T", np.amax(T))
print("max radVar1", np.amax(radVar1))
print("max radVar2", np.amax(radVar2))
# print("max Delq", np.amax(Delq))
# print("max kPlanck", np.amax(kPlanck))



InNum = 2
inputs = np.stack((T,no), axis = 1)
targets = np.stack((radVar1,radVar2), axis=1)
data = np.stack((T,no,radVar1,radVar2), axis = 1)
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.20, random_state=42, shuffle=True)

inputs =  X_train
targets = y_train
vinputs = X_test
vtargets = y_test

print("Loaded data, {} training, {} validation samples".format(len(inputs),len(vinputs)))
print("Loaded data, {} targets, {} validation targets".format(len(targets),len(vtargets)))
print("Size of the inputs array: "+format(inputs.shape))
print("Size of the targets array: "+format(vtargets.shape))
print("Size of the vinputs array: "+format(vinputs.shape))
print("Size of the targets array: "+format(targets.shape))

showSbs(inputs,targets,3)


# global training constants
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 10
LR = 0.00002

class DfpDataset():
    def __init__(self, inputs,targets): 
        self.inputs  = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

tdata = DfpDataset(inputs,targets)
vdata = DfpDataset(vinputs,targets)

trainLoader = torch.utils.data.DataLoader(tdata, batch_size=BATCH_SIZE, shuffle=True , drop_last=True) 
valiLoader  = torch.utils.data.DataLoader(vdata, batch_size=BATCH_SIZE, shuffle=False, drop_last=True) 

print("Training & validation batches: {} , {}".format(len(trainLoader),len(valiLoader) ))


def blockUNet(in_c, out_c, name, size=4, pad=1, transposed=False, bn=True, activation=True, relu=True, dropout=0. ):
    block = nn.Sequential()

    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear'))
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))

    if activation:
        if relu:
            block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        else:
            block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    return block
    
class DfpNet(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(DfpNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)
        InNum = 2
        self.layer1 = blockUNet(InNum     , channels*1, 'enc_layer1', transposed=False, bn=True, relu=False, dropout=dropout )
        self.layer2 = blockUNet(channels  , channels*2, 'enc_layer2', transposed=False, bn=True, relu=False, dropout=dropout )
        self.layer3 = blockUNet(channels*2, channels*2, 'enc_layer3', transposed=False, bn=True, relu=False, dropout=dropout )
        self.layer4 = blockUNet(channels*2, channels*4, 'enc_layer4', transposed=False, bn=True, relu=False, dropout=dropout )
        self.layer5 = blockUNet(channels*4, channels*8, 'enc_layer5', transposed=False, bn=True, relu=False, dropout=dropout ) 
        self.layer6 = blockUNet(channels*8, channels*8, 'enc_layer6', transposed=False, bn=True, relu=False, dropout=dropout , size=2,pad=0)
        self.layer7 = blockUNet(channels*8, channels*8, 'enc_layer7', transposed=False, bn=True, relu=False, dropout=dropout , size=2,pad=0)
     
        # note, kernel size is internally reduced by one for the decoder part
        self.dlayer7 = blockUNet(channels*8, channels*8, 'dec_layer7', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer6 = blockUNet(channels*16,channels*8, 'dec_layer6', transposed=True, bn=True, relu=True, dropout=dropout , size=2,pad=0)
        self.dlayer5 = blockUNet(channels*16,channels*4, 'dec_layer5', transposed=True, bn=True, relu=True, dropout=dropout ) 
        self.dlayer4 = blockUNet(channels*8, channels*2, 'dec_layer4', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer3 = blockUNet(channels*4, channels*2, 'dec_layer3', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer2 = blockUNet(channels*4, channels  , 'dec_layer2', transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer1 = blockUNet(channels*2, InNum     , 'dec_layer1', transposed=True, bn=False, activation=False, dropout=dropout )

    def forward(self, x):
        # note, this Unet stack could be allocated with a loop, of course... 
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        # ... bottleneck ...
        dout6 = self.dlayer7(out7)
        dout6_out6 = torch.cat([dout6, out6], 1)
        dout6 = self.dlayer6(dout6_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# channel exponent to control network size
EXPO = 2

# setup network
net = DfpNet(channelExponent=EXPO).to(device)
nn_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in nn_parameters])

print("Trainable params: {}  ".format(params)) 

net.apply(weights_init)
criterionL1 = nn.L1Loss()
optimizerG = optim.Adam(net.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=0.0)

targets = torch.autograd.Variable(torch.FloatTensor(BATCH_SIZE, 2, 128, 128)).to(device)
inputs  = torch.autograd.Variable(torch.FloatTensor(BATCH_SIZE, 2, 128, 128)).to(device)

#print(net) # to double check the details...
#y = net(inputs)
#make_dot(y.mean())#, params=dict(y.named_parameters()))


history_L1 = []
history_L1val = []

if os.path.isfile(dataset+'networkk'):
  print("Found existing network, loading & skipping training")
  net.load_state_dict(torch.load(dataset+'network')) # optionally, load existing network
#if 2+3 == 2:
#  print('what')
else:
  print("Training from scratch")
  for epoch in range(EPOCHS):
      net.train()
      L1_accum = 0.0
      for i, traindata in enumerate(trainLoader, 0):
          inputs_curr, targets_curr = traindata
          inputs.data.copy_(inputs_curr.float())
          targets.data.copy_(targets_curr.float())

          net.zero_grad()
          gen_out = net(inputs)

          lossL1 = criterionL1(gen_out, targets)
          lossL1.backward()
          optimizerG.step()
          L1_accum += lossL1.item()

      # validation
      net.eval()
      L1val_accum = 0.0
      for i, validata in enumerate(valiLoader, 0):
          inputs_curr, targets_curr = validata
          inputs.data.copy_(inputs_curr.float())
          targets.data.copy_(targets_curr.float())

          outputs = net(inputs)
          outputs_curr = outputs.data.cpu().numpy()

          lossL1val = criterionL1(outputs, targets)
          L1val_accum += lossL1val.item()

      # data for graph plotting
      history_L1.append( L1_accum / len(trainLoader) )
      history_L1val.append( L1val_accum / len(valiLoader) )

      if epoch<3 or epoch%20==0:
          print( "Epoch: {}, L1 train: {:7.5f}, L1 vali: {:7.5f}".format(epoch, history_L1[-1], history_L1val[-1]) )

  torch.save(net.state_dict(), dataset+'network' )
  print("Training done, saved network")


l1train = np.asarray(history_L1)
l1vali  = np.asarray(history_L1val)

plt.plot(np.arange(l1train.shape[0]),l1train,'b',label='Training loss')
plt.plot(np.arange(l1vali.shape[0] ),l1vali ,'g',label='Validation loss')
plt.legend()
plt.show()


net.eval()
for i, validata in enumerate(valiLoader, 0):
    inputs_curr, targets_curr = validata
    inputs.data.copy_(inputs_curr.float())
    targets.data.copy_(targets_curr.float())
    
    outputs = net(inputs)
    outputs_curr = outputs.data.cpu().numpy()
    #if i<1: showSbs(targets_curr , outputs_curr,i)


# Test on new distribution

nond,Tnd,radVar1nd,radVar2nd,Delqnd,kPlancknd = getdata(dataset+'test/')
znond,zTnd,zradVar1nd,zradVar2nd,zDelqnd,zkPlancknd = nond,Tnd,radVar1nd,radVar2nd,Delqnd,kPlancknd
nond,Tnd,Delqnd,kPlancknd = getnorm(nond,Tnd,radVar1nd,radVar2nd,Delqnd,kPlancknd,'input')
radVar1nd,radVar2nd = getnorm(nond,Tnd,radVar1nd,radVar2nd,Delqnd,kPlancknd,'target')

     
testinputs = np.stack((Tnd,nond), axis = 1)
testtargets = np.stack((radVar1nd,radVar2nd), axis=1)

testdata = DfpDataset(testinputs,testtargets)
testLoader  = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False, drop_last=True) 

net.eval()
L1t_accum = 0.
testresultA = np.zeros((ns,nps,nps))
testresultE = np.zeros((ns,nps,nps))
for i, validata in enumerate(testLoader, 0):
    inputs_curr, targets_curr = validata
    inputs.data.copy_(inputs_curr.float())
    targets.data.copy_(targets_curr.float())

    outputs = net(inputs)
    outputs_curr = outputs.data.cpu().numpy()
    o = outputs_curr[0]
    testresultA[i,:,:] = o[0,:,:]
    testresultE[i,:,:] = o[1,:,:]
    lossL1t = criterionL1(outputs, targets)
    L1t_accum += lossL1t.item()
    #if i%100 ==0: showSbs(targets_curr,outputs_curr,0)

print("\nAverage test error: {}".format( L1t_accum/len(testLoader) ))


nond,Tnd,Delqnd,kPlancknd= getdenorm(znond,zTnd,zradVar1nd,zradVar2nd,zDelqnd,zkPlancknd,nond,Tnd,radVar1nd,radVar2nd,Delqnd,kPlancknd,'input')
radVar1nd,radVar2nd = getdenorm(zno,zT,zradVar1nd,zradVar2nd,zDelqnd,zkPlancknd,nond,Tnd,radVar1nd,radVar2nd,Delqnd,kPlancknd,'target')
getani(nond,Tnd,radVar1nd,radVar2nd,Delqnd,kPlancknd,ns,dataset,'nd',norm=False)


radVar1test,radVar2test = getdenorm(znond,zTnd,zradVar1nd,zradVar2nd,zDelqnd,zkPlancknd,nond,Tnd,testresultA,testresultE,Delqnd,kPlancknd,'target')
getani(nond,Tnd,radVar1test,radVar2test,Delqnd,kPlancknd,ns,dataset,'testresults',norm=False)


radVar1ErrorAbs = np.zeros((ns,nps,nps))
radVar2ErrorAbs = np.zeros((ns,nps,nps))

for i in range(ns):
    radVar1ErrorAbs[i,:,:] = np.abs(radVar1nd[i,:,:] - radVar1test[i,:,:])
    radVar2ErrorAbs[i,:,:] = np.abs(radVar2nd[i,:,:] - radVar2test[i,:,:])

getani(nond,Tnd,radVar1ErrorAbs,radVar2ErrorAbs,Delqnd,kPlancknd,ns,dataset,'errorAbs',norm=False)
    

