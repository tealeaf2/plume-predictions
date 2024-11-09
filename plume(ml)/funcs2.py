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
from scipy.io import savemat

#All from previous CNN functions
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.eps = eps

    #Use this for normalization
    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    #Use this for denorm
    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        x = (x * std) + mean
        return x



def ani_frame(X, N_TIME_STEPS, name, dataset, norm=False, dpi=100):
    # Create a figure with a single subplot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(X[5,:,:], origin='lower', interpolation='bicubic', cmap='jet')
    fig.colorbar(im)

    test = X[X!=0]
    print('mean/stddev', np.mean(test), np.std(test))
    
    im.set_clim([0, np.amax(X)])

    def update_img(n):
        tmp = X[n,:,:]
        im.set_data(tmp)
        return im

    ani = animation.FuncAnimation(fig, update_img, N_TIME_STEPS, interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save(dataset + 'cnn/' + name + '_demo.mp4', writer=writer, dpi=dpi)
    return ani

def getani(X, ns, dataset, which, norm=False):
    ani_frame(X, ns, which, dataset, norm)

    
def loaddata(dataset, var):
    varz = np.load(dataset + var + '.npz')
    var = np.transpose(varz['arr_0'], (2, 0, 1))
    return var

def getdata(dataset):
    variables = ['no', 'T', 'kPlanck','radVar1', 'radVar2', 'Delq']
    data = {}
    for i, var in enumerate(variables):
        data[var] = loaddata(dataset, var)
    return data

def organizedata(dataset,which):
    data = getdata(dataset+which)
    zdata = data

    ns = len(data['no'][:, 0, 0])
    nps = len(data['no'][0, :, 0])

    print("number of time steps, ns: ", ns)
    print("number pixels, nps: ", nps)
    print("max no", np.amax(data['no']))
    print("max T", np.amax(data['T']))
    print("max radVar1", np.amax(data['radVar1']))
    print("max radVar2", np.amax(data['radVar2']))

    input_vars = []
    target_vars = []
    for var in data:
        if var == 'no' or var == 'T':
            input_vars.append(data[var])
        if var == 'radVar1' or var == 'radVar2':
            target_vars.append(data[var])

    # Stack the input and target variables into arrays
    inputs = np.stack(input_vars, axis=1)
    targets = np.stack(target_vars, axis=1)
    return inputs, targets, zdata



def showSbs(inputs, targets, i):
    f, ax = plt.subplots(2, 2)
    f.tight_layout()
    [axi.set_axis_off() for axi in ax.ravel()]
    
    data = [(inputs[i, 0, :, :], 'Input 1'), (inputs[i, 1, :, :], 'Input 2'),
            (targets[i, 0, :, :], 'Target 1'), (targets[i, 1, :, :], 'Target 2')]
    
    for axi, (data, label) in zip(ax.ravel(), data):
        im = axi.imshow(data, origin='lower', cmap='jet')
        im.set_clim([0, np.amax(data)])
        axi.set_title(label)
    
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im, cax=cbar_ax)
    
    