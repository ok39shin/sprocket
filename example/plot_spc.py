#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# 数値計算用のnumpyモジュールをインポートし,モジュール名をnpとする
import numpy as np

import matplotlib
matplotlib.use('Agg')
# グラフを描画するためにmatplotlibライブラリからpyplotをインポート
from matplotlib import pyplot as plt

from sprocket.util import HDF5 

# read h5
jntf = os.path.join('data/pair/SF1-TF1', 'jnt', 'it' + '3' + '_jnt.h5')
jnth5 = HDF5(jntf, mode='r')
jnt = jnth5.read(ext='mcep')
# fig = plt.figure()
for i in range(4):
    plt.subplot(4,1,i+1)
    plt.imshow(jnt.T[24*i:24*(i+1),:],  
               aspect="auto", origin="lower", cmap="jet"
               )

plt.savefig('spc.png')

