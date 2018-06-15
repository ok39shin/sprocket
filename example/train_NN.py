" training model by Neural Network "
# coding : utf-8

# for some
import sys      # for test sys.exit
import os       # for os.path.join
import torch    # for use pytorch

# for load_data
from sprocket.util import HDF5 # treat h5 file

# for make_DataLoader
from sklearn.model_selection import train_test_split    # devide data train and dev
from torch.utils.data import TensorDataset, DataLoader  # 

# for load_Model
from src import models

# for set loss and optim function
from torch import nn
from torch import optim 

# for training
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available() # flag of using cuda 

def load_data(pair_dir):
    """
    load data

    input   : pair_dir -> jnt.h5 file dir
    output  : jnt_src_mcep, jnt_tar_mcep -> jointed mcep src and tar

    """
    # read joint feature vector
    jntf = os.path.join(pair_dir, 'jnt',
                        'it' + '3' + '_jnt.h5')
    #jntf = os.path.join(args.pair_dir, 'jnt',
    #                    'it' + str(pconf.jnt_n_iter) + '_jnt.h5')
    
    jnth5 = HDF5(jntf, mode='r')
    jnt = jnth5.read(ext='mcep')
    jntD1 = jnt.shape[1]
    jnt_src_mcep = jnt[:,:jntD1//4]                 # src static
    # jnt_src_delta = jnt[:,jntD1//4:jntD1//4*2]    # src delta
    jnt_tar_mcep = jnt[:,jntD1//4*2:jntD1//4*3]     # tar static
    # jnt_tar_delta = jnt[jntD1//4*3:]              # tar delta
    return jnt_src_mcep, jnt_tar_mcep

def make_DataLoader(src_mcep, tar_mcep):
    """ 
    make DataLoader 

    input   : src_mcep, tar_mcep -> jointed mcep src and tar
    output  : loader_train, loader_test -> train src tar, test src tar

    """
    # split data to train and dev 
    src_train, src_dev, tar_train, tar_dev = \
            train_test_split(src_mcep, tar_mcep,
                             test_size=1/10,
                             # random_state=0 # for study
                             )
    
    # transform np.ndarray to torch.Tensor
    src_train = torch.Tensor(src_train)
    src_dev = torch.Tensor(src_dev)
    tar_train = torch.Tensor(tar_train)
    tar_dev = torch.Tensor(tar_dev)
    if use_cuda:
        # torch.Tensor -> torch.cuda.Tensor
        src_train = src_train.cuda()
        src_dev = src_dev.cuda()
        tar_train = tar_train.cuda()
        tar_dev = tar_dev.cuda()
    
    # make Dataset src_mcep and tar_mcep
    dataset_train = TensorDataset(src_train, tar_train)
    dataset_dev = TensorDataset(src_dev, tar_dev)
    # print(type(dataset_train))
    
    # make DataLoader
    loader_train = DataLoader(dataset_train, # data set
                              batch_size=64, # size of minibatch
                              shuffle=True   # do shuffle each epoch
                              )
    loader_dev = DataLoader(dataset_dev,
                            batch_size=32,
                            shuffle=False
                            )
    return loader_train, loader_dev

def load_Model():
    """
    load Model from src/Model.py 
   
    input   : No
    output  : model -> src.models.SimpleNN
    
    """
    #    # Define and Run (like Keras)
    #    model = nn.Sequential()
    #    model.add_module('fc1', nn.Linear(24, 12))
    #    model.add_module('relu', nn.ReLU())
    #    model.add_module('fc2', nn.Linear(12, 24))
    model = models.SimpleNN()
    if use_cuda:
        model = model.cuda()
    # print(model)
    return model

def set_LossFandOptimF(model):
    """
    settings of LossF OptimF 
    
    input   : No
    output  : lo

    """
    # setting Loss Function
    criterion = nn.MSELoss() # check: change to use test loss
    
    # setting Optim Function
    optimizer = optim.Adam(model.parameters(), lr=0.01) # lr = study rate
    return criterion, optimizer


# setting of Training
def train(model, criterion, optimizer, 
          loader_train, loader_dev, total_epoch):
    """ 
    settings of Training and Generation 
    
    input   : 
              model         -> Network model
              criterion     -> loss function
              optimizer     -> optimize function
              loader_train  -> train data set
              epoch         -> number of through all data

    output  : loss  -> loss of loss function
    
    """
    savef = os.path.join('my_model', 'second_NN.mdl')
    minloss = 1.0
    for epoch in range(total_epoch):
        model.train() # change mode to Train
        # train
        trloss = []
        for src, tar in loader_train:
            src, tar = Variable(src), Variable(tar) # enable to bibun
            optimizer.zero_grad()   # reset result of cal grad
            estm_tar = model(src)   # src into model and get estimated tar
            loss = criterion(estm_tar, tar)
            loss.backward()         # cal back propagation of loss
            optimizer.step()        # update parameter
            trloss.append(loss.item())
        
        model.eval() # change mode to Eval
        # cal loss of dev data
        devloss = []
        for src, tar in loader_dev:
            src, tar = Variable(src), Variable(tar) # enable to bibun
            optimizer.zero_grad()   # reset result of cal grad
            estm_tar = model(src)   # src into model and get estimated tar
            loss = criterion(estm_tar, tar)
            devloss.append(loss.item())
    
        print('epoch [%d/%d] Finish, TrLoss: %.5f, DevLoss: %.5f' %
                (epoch, total_epoch, np.mean(trloss), np.mean(devloss)))
        
        # save model if devloss to be smaller
        if minloss > np.mean(devloss):
            minloss = np.mean(devloss)
            torch.save(model.state_dict(), savef)
    
### run Training and Save model ###
def main():
    pair_dir = 'data/pair/SF1-TF1'
    total_epoch = 20
    # load data
    src_mcep, tar_mcep = load_data(pair_dir)
    # get data loader
    loader_train, loader_dev = make_DataLoader(src_mcep, tar_mcep)
    # get model
    model = models.SimpleNN()
    if use_cuda:
        model = model.cuda()
    # get loss and optimize function
    criterion, optimizer = set_LossFandOptimF(model)
    # train
    train(model, criterion, optimizer, 
          loader_train, loader_dev, total_epoch)
    print('Complete Training!')

if __name__ == '__main__':
    main()
