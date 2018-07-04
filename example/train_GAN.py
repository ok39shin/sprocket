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

class model_param():
    def __init__(self, mdl, optm, steps, mdlf):
        """
        input:
              mdl   -> Network model 
              crit  -> loss function 
              optm  -> optimize function 
              steps -> training steps 
              mdlf  -> model file 
        """
        self.mdl = mdl
        self.optm = optm
        self.steps = steps
        self.mdlf = mdlf

# setting of Training
def train(g, d,
          loader_train, loader_dev, epoch):
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
    # warm up G
    print('Warm Up Generator')
    train_g_warmup(g, loader_train, loader_dev, epoch)

    # train GAN
    print('Training GAN')
    train_d_g(g, d, loader_train, loader_dev, epoch)

def train_g_warmup(g, loader_train, loader_dev, epoch):
    # Training Discriminator on real + fake
    minloss_g = 100.0
    best_g = copy.copy(g.mdl) # copy now best g
    criterion = nn.MSELoss()

    for e in range(epoch):
        print('[%d/%d] Warm Up Generator' % (e+1, epoch))
        # train Generator
        g.mdl.train()  # change mode to Train
        tr_loss_g = []
        for src, tar in loader_train:
            # train G
            src, tar = Variable(src), Variable(tar)
            g.mdl.zero_grad()
            fake_tar = g.mdl(src)
            loss_mse = criterion(fake_tar, tar)
            loss_mse.backward()
            g.optm.step()
            tr_loss_g.append(loss_mse.item())
        
        # develop Generator
        g.mdl.eval()  # change mode to Train
        dev_loss_g = []
        for src, tar in loader_dev:
            # cal loss of dev 
            src, tar = Variable(src), Variable(tar)
            fake_tar = g.mdl(src)
            loss_mse = criterion(fake_tar, tar)
            dev_loss_g.append(loss_mse.item())
        
        print("TrLossG: %.5f, DevLossG: %.5f" %
              (np.mean(tr_loss_g), np.mean(dev_loss_g)))
        
        # save model if devloss to be smaller
        dev_loss = np.mean(dev_loss_g)/2
        if minloss_g > dev_loss:
            minloss_g = dev_loss
            best_g = copy.copy(g.mdl)
            torch.save(g.mdl.state_dict(), g.mdlf)
            print('Save Generator: ', g.mdlf)
        # forend d.steps
        g.mdl = best_g

def train_d_g(g, d, loader_train, loader_dev, epoch):
    for e in range(epoch):
        print('Epoch [%d/%d]' % (e+1, epoch))
        # Train Discriminator
        train_d(g, d, loader_train, loader_dev)
        # Train Generator
        train_g(g, d, loader_train, loader_dev)

def train_d(g, d, loader_train, loader_dev):
    # Training Discriminator on real + fake
    g.mdl.eval()  # change mode to Eval
    minloss_d = 100.0
    best_d = copy.copy(d.mdl) # copy now best d
    eps = 1e-20
    criterion = nn.BCELoss()
    for steps in range(d.steps):
        d.mdl.train() # change mode to Train
        print('Discriminator [%d/%d]' % (steps, d.steps))
        tr_real_loss_d = []
        tr_fake_loss_d = []
        for src, tar in loader_train:
            # prepare labels
            variable1 = Variable(torch.ones((src.shape[0], 1)))
            variable0 = Variable(torch.zeros((src.shape[0], 1)))
            if use_cuda:
                variable1 = variable1.cuda()
                variable0 = variable0.cuda()

            # train on real
            tar = Variable(tar)             # enable to bibun
            d.optm.zero_grad()        # reset result of cal grad
            real_d = d.mdl(tar)   # D estimate real value
            loss_real_d = criterion(real_d, variable1)

            # train on fake from G
            src = Variable(src)
            fake_tar = g.mdl(src)         # fake from G
            fake_d = d.mdl(fake_tar)  # D estimate fake value
            loss_fake_d = criterion(fake_d, variable0)
   
            # Loss
            loss_d = loss_real_d + loss_fake_d

            tr_real_loss_d.append(loss_real_d.item())
            tr_fake_loss_d.append(loss_fake_d.item())

            loss_d.backward()
            d.optm.step()      # update param
        # train end

        d.mdl.eval()
        # cal loss of dev data
        dev_real_loss_d = []
        dev_fake_loss_d = []
        for src, tar in loader_dev:
            variable1 = Variable(torch.ones((src.shape[0], 1)))
            variable0 = Variable(torch.zeros((src.shape[0], 1)))
            if use_cuda:
                variable1 = variable1.cuda()
                variable0 = variable0.cuda()
            # cal loss from develop real data
            tar = Variable(tar)             # enable to bibun
            d.optm.zero_grad()        # reset result of cal grad
            real_d = d.mdl(tar)   # D estimate real value
            loss_real_d = criterion(real_d, variable1)

            # cal loss from develop fake data from G
            src = Variable(src)
            fake_tar = g.mdl(src)         # fake from G
            fake_d = d.mdl(fake_tar)  # D estimate fake value
            loss_fake_d = criterion(fake_d, variable0)

            # Loss
            dev_real_loss_d.append(loss_real_d.item())
            dev_fake_loss_d.append(loss_fake_d.item())
        # eval end

        print('''TrRealLoss: %.5f, TrFakeLoss: %.5f
DevRealLoss: %.5f, DevFakeLoss: %.5f''' %
              (np.mean(tr_real_loss_d), np.mean(tr_fake_loss_d),
               np.mean(dev_real_loss_d), np.mean(dev_fake_loss_d)))
        
        # save # model if devloss to be smaller
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
    eps = 1e-20
    criterion1 = nn.MSELoss()
    criterion2 = nn.BCELoss()
    for steps in range(g.steps):
        print('Generator [%d/%d]' % (steps, g.steps))
        g.mdl.train()  # change mode to Train
        tr_loss_g = []
        tr_loss_from_d = []
        for src, tar in loader_train:
            variable1 = Variable(torch.ones((src.shape[0], 1)))
            if use_cuda:
                variable1 = variable1.cuda()
            # train G
            src, tar = Variable(src), Variable(tar)
            g.mdl.zero_grad()
            fake_tar = g.mdl(src)
            loss_mse = criterion1(fake_tar, tar)

            fake_d = d.mdl(fake_tar)
            loss_from_d = criterion2(fake_d, variable1)

            # Loss
            loss_g = loss_mse + loss_from_d

            tr_loss_g.append(loss_mse.item())
            tr_loss_from_d.append(loss_from_d.item())

            loss_g.backward()
            g.optm.step()
    
        g.mdl.eval()  # change mode to Train
        dev_loss_g = []
        dev_loss_from_d = []
        for src, tar in loader_dev:
            variable1 = Variable(torch.ones((src.shape[0], 1)))
            if use_cuda:
                variable1 = variable1.cuda()
            # cal loss of dev 
            src, tar = Variable(src), Variable(tar)
            fake_tar = g.mdl(src)
            loss_mse = criterion1(fake_tar, tar)

            fake_d = d.mdl(fake_tar)
            loss_from_d = criterion2(fake_d, variable1)

            dev_loss_g.append(loss_mse.item())
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
    total_epoch = 10
    steps_g = 1
    steps_d = 1
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
    # setting Optim Function
    optimizer_g = optim.Adam(model_g.parameters(), lr=0.01) # lr = study rate
    optimizer_d = optim.Adam(model_d.parameters(), lr=0.01)

    mdlf_g = os.path.join('my_model','gan_g.mdl')
    mdlf_d = os.path.join('my_model','gan_d.mdl')
    model_g_param = model_param(model_g,
                                optimizer_g,
                                steps_g,
                                mdlf_g)
    model_d_param = model_param(model_d,
                                optimizer_d,
                                steps_d,
                                mdlf_d)

    # train
    train(model_g_param, model_d_param, 
          loader_train, loader_dev, total_epoch)
    print('Complete Training!')

if __name__ == '__main__':
    main()
