" training model by Neural Network "
# coding : utf-8
import sys
import os
from sprocket.util import HDF5
import torch

use_cuda = torch.cuda.is_available()

### load data ###

# read joint feature vector
jntf = os.path.join('data/pair/SF1-TF1', 'jnt',
                    'it' + '3' + '_jnt.h5')
#jntf = os.path.join(args.pair_dir, 'jnt',
#                    'it' + str(pconf.jnt_n_iter) + '_jnt.h5')
jnth5 = HDF5(jntf, mode='r')
jnt = jnth5.read(ext='mcep')
print('jnt.shape : ',jnt.shape)
jntD1 = jnt.shape[1]
jnt_src = jnt[:,:jntD1//4] # src static
# jnt_src_delta = jnt[:,jntD1//4:jntD1//4*2] src delta
jnt_tar = jnt[:,jntD1//4*2:jntD1//4*3] # tar static
# jnt_tar_delta = jnt[jntD1//4*3:] tar delta
print(jnt_src.shape)
print(jnt_tar.shape)
print(type(jnt_src))
# sys.exit(1)

### make Dataloder ###
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
# split data to train and test
src_train, src_test, tar_train, tar_test = \
        train_test_split(jnt_src, jnt_tar,
                         test_size=1/10,
                         random_state=0)
print(src_train.shape)
print(src_test.shape)

# transform np.ndarray to torch.Tensor
src_train = torch.Tensor(src_train)
src_test = torch.Tensor(src_test)
tar_train = torch.Tensor(tar_train)
tar_test = torch.Tensor(tar_test)
print(type(src_train))
print(src_train.shape)
if use_cuda:
    src_train = src_train.cuda()
    src_test = src_test.cuda()
    tar_train = tar_train.cuda()
    tar_test = tar_test.cuda()

# make DataSet src_mcep and tar_mcep
dataset_train = TensorDataset(src_train, tar_train)
dataset_test = TensorDataset(src_test, tar_test)
print(type(dataset_train))
# print(dataset_train.shape)

# make DataLoader
loader_train = DataLoader(dataset_train, # data set
                          batch_size=64, # size of minibatch
                          shuffle=True   # do shuffle each epoch
                          )
loader_test = DataLoader(dataset_test, 
                          batch_size=32,
                          shuffle=False
                          )

### establish Network ###
from torch import nn

# Define and Run (like Keras)
model = nn.Sequential()
model.add_module('fc1', nn.Linear(24, 12))
model.add_module('relu', nn.ReLU())
model.add_module('fc2', nn.Linear(12, 24))
if use_cuda:
    model = model.cuda()
print(model)

### settings of LossF OptimF ###
from torch import optim 

# setting Loss Function
criterion = nn.MSELoss() # check: change to use test loss

# setting Optim Function
optimizer = optim.Adam(model.parameters(), lr=0.01) # lr = study rate

### settings of Training and Generation ###
from torch.autograd import Variable

# setting of Training
def train(epoch):
    model.train() # change mode to Train
    for src, tar in loader_train:
        src, tar = Variable(src), Variable(tar) # enable to bibun
        optimizer.zero_grad() # reset result of cal grad
        estm_tar = model(src) # src into model and get estimated tar
        loss = criterion(estm_tar, tar)
        loss.backward() # cal back propagation of loss
        optimizer.step()
    print("epoch{}: finish\n".format(epoch))
    print('loss:', loss)

# setting of Generation
#def generation():
#    model.eval()
#    for src, tar in loader_test:
#        src, tra = Variable(src), Variable(tar)
#        estm_tar = model(src)
    

### run Training and Save model ###
for epoch in range(20):
    train(epoch)
print('Complete Training!')
save = os.path.join('my_model', 'first_NN.mdl')
torch.save(model.state_dict(), save)
print('model save to :', save)

