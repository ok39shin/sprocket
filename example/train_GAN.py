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
import copy

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

class model_param():
    def __init__(self, mdl, crit, optm, steps, mdlf):
        """
        input:
              mdl   -> Network model 
              crit  -> loss function 
              optm  -> optimize function 
              steps -> training steps 
              mdlf  -> model file 
        """
        self.mdl = mdl
        self.crit = crit
        self.optm = optm
        self.steps = steps
        self.mdlf = mdlf

# setting of Training
def train(g, d,
          loader_train, loader_dev, total_epoch):
    """ 
    settings of Training Generator and Discriminator
    
    input   : 
              g -> G param
              d -> D param
              loader_train  -> train data set
              loader_dev    -> develop data set
              epoch         -> number of through all data

    output  : loss  -> loss of loss function
    
    """
    for epoch in range(total_epoch):
        print('Epoch [%d/%d]' % (epoch+1, total_epoch))
        # Train Discriminator
        train_d(g, d, loader_train, loader_dev)
        # Train Generator
        train_g(g, d, loader_train, loader_dev)
        print()

def train_d(g, d, loader_train, loader_dev):
    # Training Discriminator on real + fake
    g.mdl.eval()  # change mode to Eval
    tr_real_loss_d = []
    tr_fake_loss_d = []
    minloss_d = 1.0
    best_d = copy.copy(d.mdl) # copy now best d
    for steps in range(d.steps):
        d.mdl.train() # change mode to Train
        print('Discriminator [%d/%d]' % (steps, d.steps))
        for src, tar in loader_train:
            # prepare for train
            variable1 = Variable(torch.ones(src.shape[0], 1))  # for train real
            variable0 = Variable(torch.zeros(src.shape[0], 1)) # for train fake
            if use_cuda:
                variable1 = variable1.cuda()
                variable0 = variable0.cuda()
            # train on real
            tar = Variable(tar)             # enable to bibun
            d.optm.zero_grad()        # reset result of cal grad
            real_dic_d = d.mdl(tar)   # D estimate real value
            real_loss_d = d.crit(real_dic_d, variable1) # loss from real
            real_loss_d.backward()         # cal back propagation of loss
            tr_real_loss_d.append(real_loss_d.item())
    
            # train on fake from G
            src = Variable(src)
            fake_tar = g.mdl(src)         # fake from G
            fake_dic_d = d.mdl(fake_tar)  # D estimate fake value
            fake_loss_d = d.crit(fake_dic_d, variable0) # loss from fake
            fake_loss_d.backward()
            tr_fake_loss_d.append(fake_loss_d.item())
    
            d.optm.step()      # update param
        
        d.mdl.eval()
        # cal loss of dev data
        dev_real_loss_d = []
        dev_fake_loss_d = []
        for src, tar in loader_dev:
            # prepare for train
            variable1 = Variable(torch.ones(src.shape[0], 1))  # for train real
            variable0 = Variable(torch.zeros(src.shape[0], 1)) # for train fake
            if use_cuda:
                variable1 = variable1.cuda()
                variable0 = variable0.cuda()
            # cal loss from develop real data
            tar = Variable(tar)             # enable to bibun
            d.optm.zero_grad()        # reset result of cal grad
            real_dic_d = d.mdl(tar)   # D estimate real value
            real_loss_d = d.crit(real_dic_d, variable1) # loss from real
            dev_real_loss_d.append(real_loss_d.item())
    
            # cal loss from develop fake data from G
            src = Variable(src)
            fake_tar = g.mdl(src)         # fake from G
            fake_dic_d = d.mdl(fake_tar)  # D estimate fake value
            fake_loss_d = d.crit(fake_dic_d, variable0) # loss from fake
            dev_fake_loss_d.append(fake_loss_d.item())
    
        print('''TrRealLoss: %.5f, TrFakeLoss: %.5f
DevRealLoss: %.5f, DevFakeLoss: %.5f''' %
              (np.mean(tr_real_loss_d), np.mean(tr_fake_loss_d),
               np.mean(dev_real_loss_d), np.mean(dev_fake_loss_d)))
        
        # save model if devloss to be smaller
        dev_loss = (np.mean(dev_real_loss_d)+np.mean(dev_fake_loss_d))/2
        if minloss_d > dev_loss:
            minloss_d = dev_loss
            best_d = copy.copy(d.mdl)
    # forend d.steps
    d.mdl = best_d
    torch.save(d.mdl.state_dict(), d.mdlf)
    print('Save Discriminator: ', d.mdlf)

def train_g(g, d, loader_train, loader_dev):
    # Training Discriminator on real + fake
    d.mdl.eval()   # change mode to Eval 
    minloss_g = 1.0
    best_g = copy.copy(g.mdl) # copy now best g
    for steps in range(g.steps):
        print('Generator [%d/%d]' % (steps, g.steps))
        g.mdl.train()  # change mode to Train
        tr_loss_g = []
        tr_loss_from_d = []
        for src, tar in loader_train:
            # prepare for train
            variable1 = Variable(torch.ones(src.shape[0], 1))  # for train real
            if use_cuda:
                variable1 = variable1.cuda()
            # train G
            src, tar = Variable(src), Variable(tar)
            g.mdl.zero_grad()
            fake_tar = g.mdl(src)
            loss_g = g.crit(fake_tar, tar) # loss from tar
            tr_loss_g.append(loss_g.item())
            loss_g.backward(retain_graph=True)

            fake_dic_d = d.mdl(fake_tar)
            loss_from_d = g.crit(fake_dic_d, variable1) # loss from D
            tr_loss_from_d.append(loss_from_d.item())
            loss_from_d.backward() 

            g.optm.step()
    
        g.mdl.eval()  # change mode to Train
        dev_loss_g = []
        dev_loss_from_d = []
        for src, tar in loader_dev:
            # prepare
            variable1 = Variable(torch.ones(src.shape[0], 1))  # for train real
            if use_cuda:
                variable1 = variable1.cuda()
            # cal loss of dev 
            src, tar = Variable(src), Variable(tar)
            fake_tar = g.mdl(src)
            loss_g = g.crit(fake_tar, tar) # loss from tar
            dev_loss_g.append(loss_g.item())

            fake_dic_d = d.mdl(fake_tar)
            loss_from_d = g.crit(fake_dic_d, variable1) # loss from D
            dev_loss_from_d.append(loss_from_d.item())

        print('''TrLossG: %.5f, TrLossfromD: %.5f
DevLossG: %.5f, DevLossfromD: %.5f''' %
              (np.mean(tr_loss_g), np.mean(tr_loss_from_d),
               np.mean(dev_loss_g), np.mean(dev_loss_from_d)))
        
        # save model if devloss to be smaller
        dev_loss = (np.mean(dev_loss_g)+np.mean(dev_loss_from_d))/2
        if minloss_g > dev_loss:
            minloss_g = dev_loss
            best_g = copy.copy(g.mdl)
    # forend d.steps
    g.mdl = best_g
    torch.save(g.mdl.state_dict(), g.mdlf)
    print('Save Generator: ', g.mdlf)

### run Training and Save model ###
def main():
    pair_dir = 'data/pair/SF1-TF1'
    total_epoch = 20
    steps_g = 10
    steps_d = 5
    # load data
    src_mcep, tar_mcep = load_data(pair_dir)
    # get data loader
    loader_train, loader_dev = make_DataLoader(src_mcep, tar_mcep)
    # get Generater model, Discriminater model
    model_g = models.SimpleGene()
    model_d = models.SimpleDisc()
    if use_cuda:
        model_g = model_g.cuda()
        model_d = model_d.cuda()

    # get loss and optimize function
    criterion_g, optimizer_g = set_LossFandOptimF(model_g)
    criterion_d, optimizer_d = set_LossFandOptimF(model_d)

    mdlf_g = os.path.join('my_model','gan_g.mdl')
    mdlf_d = os.path.join('my_model','gan_d.mdl')
    model_g_param = model_param(model_g,
                                criterion_g,
                                optimizer_g,
                                steps_g,
                                mdlf_g)
    model_d_param = model_param(model_d,
                                criterion_d,
                                optimizer_d,
                                steps_d,
                                mdlf_d)

    # train
    train(model_g_param, model_d_param, 
          loader_train, loader_dev, total_epoch)
    print('Complete Training!')

if __name__ == '__main__':
    main()
