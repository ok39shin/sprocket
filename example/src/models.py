"how to use model"
# coding : utf-8
import torch
from torch import nn
import os
import torch.nn.functional as F

def load_model(mdl_path):
    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(24, 12))
    model.add_module('relu', nn.ReLU())
    model.add_module('fc2', nn.Linear(12, 24))
    model.load_state_dict(torch.load(mdl_path))
    return model

class SimpleNN(nn.Module):
    " Most Simple Neural Network Model "
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(24, 12)
        self.fc2 = nn.Linear(12, 24)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    model_path = os.path.join('my_model', 'first_NN.mdl')
    model = load_model(model_path)
    
    print(model)
