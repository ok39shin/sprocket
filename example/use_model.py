"how to use model"
# coding : utf-8
import torch
from torch import nn
import os

def load_model(mdl_path):
    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(48, 24))
    model.add_module('relu', nn.ReLU())
    model.add_module('fc2', nn.Linear(24, 48))
    model.load_state_dict(torch.load(mdl_path))
    return model

if __name__ == '__main__':
    model_path = os.path.join('my_model', 'first_NN.mdl')
    model = load_model(model_path)
    
    print(model)
