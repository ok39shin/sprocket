#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conversion

"""

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile
from sklearn.externals import joblib

from sprocket.model import GV, F0statistics, GMMConvertor
from sprocket.speech import FeatureExtractor, Synthesizer
from sprocket.util import HDF5, static_delta

from .misc import low_cut_filter
from .yml import PairYML, SpeakerYML

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

import models

use_cuda = torch.cuda.is_available()

def main(*argv):
    argv = argv if argv else sys.argv[1:]
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-gmmmode', '--gmmmode', type=str, default=None,
                        help='mode of the GMM [None, diff, or intra]')
    parser.add_argument('org', type=str,
                        help='Original speaker')
    parser.add_argument('tar', type=str,
                        help='Target speaker')
    parser.add_argument('org_yml', type=str,
                        help='Yml file of the original speaker')
    parser.add_argument('pair_yml', type=str,
                        help='Yml file of the speaker pair')
    parser.add_argument('eval_list_file', type=str,
                        help='List file for evaluation')
    parser.add_argument('wav_dir', type=str,
                        help='Directory path of source spekaer')
    parser.add_argument('pair_dir', type=str,
                        help='Directory path of pair directory')
    args = parser.parse_args(argv)

    # read parameters from speaker yml
    sconf = SpeakerYML(args.org_yml)
    pconf = PairYML(args.pair_yml)

    # read NN for mcep
    model = models.SimpleNN() # Simple NN model
    mdl_path = os.path.join('my_model', 'second_NN.mdl')
    model.load_state_dict(torch.load(mdl_path))
    if use_cuda:
        model = model.cuda()
    model.eval() # not need

    # read F0 statistics
    stats_dir = os.path.join(args.pair_dir, 'stats')
    orgstatspath = os.path.join(stats_dir,  args.org + '.h5')
    orgstats_h5 = HDF5(orgstatspath, mode='r')
    orgf0stats = orgstats_h5.read(ext='f0stats')
    orgstats_h5.close()

    # read F0 and GV statistics for target
    tarstatspath = os.path.join(stats_dir,  args.tar + '.h5')
    tarstats_h5 = HDF5(tarstatspath, mode='r')
    tarf0stats = tarstats_h5.read(ext='f0stats')
    tarstats_h5.close()

    f0stats = F0statistics()

    # constract FeatureExtractor class
    feat = FeatureExtractor(analyzer=sconf.analyzer,
                            fs=sconf.wav_fs,
                            fftl=sconf.wav_fftl,
                            shiftms=sconf.wav_shiftms,
                            minf0=sconf.f0_minf0,
                            maxf0=sconf.f0_maxf0)

    # constract Synthesizer class
    synthesizer = Synthesizer(fs=sconf.wav_fs,
                              fftl=sconf.wav_fftl,
                              shiftms=sconf.wav_shiftms)

    # test directory
    test_dir = os.path.join(args.pair_dir, 'test')
    os.makedirs(os.path.join(test_dir, args.org), exist_ok=True)

    # conversion in each evaluation file
    with open(args.eval_list_file, 'r') as fp:
        for line in fp:
            # open wav file
            f = line.rstrip()
            wavf = os.path.join(args.wav_dir, f + '.wav')
            fs, x = wavfile.read(wavf)
            x = x.astype(np.float)
            x = low_cut_filter(x, fs, cutoff=70)
            assert fs == sconf.wav_fs

            # analyze F0, mcep, and ap
            f0, _, ap = feat.analyze(x)
            mcep = feat.mcep(dim=sconf.mcep_dim, alpha=sconf.mcep_alpha)
            mcep_0th = mcep[:, 0] # float64

            # convert F0
            cvf0 = f0stats.convert(f0, orgf0stats, tarf0stats)

            mcep_Tnsr = torch.Tensor(mcep[:,1:])
            if use_cuda:
                mcep_Tnsr = mcep_Tnsr.cuda()
            mcep_Tnsr = Variable(mcep_Tnsr) # to be bibun kanou
            cvmcep_wopow = model(mcep_Tnsr)
            if use_cuda:
                cvmcep_wopow = cvmcep_wopow.data.cpu().numpy()
            else:
                cvmcep_wopow = cvmcep_wopow.data.numpy()
            cvmcep_wopow = cvmcep_wopow.astype('float64')

            # get cvmcep [mcep0th(T, 1), cvmcep_wopow(T, 24)]
            cvmcep = np.insert(cvmcep_wopow, 0, 
                               mcep_0th, 
                               axis=1)

           # synthesis VC w/ GV
            if args.gmmmode is None:
                wav = synthesizer.synthesis(cvf0,
                                            cvmcep,
                                            ap,
                                            rmcep=mcep,
                                            alpha=sconf.mcep_alpha,
                                            )
                wavpath = os.path.join(test_dir, f + '_GAN_VC.wav')

            # synthesis DIFFVC w/ GV
            # disable
            if args.gmmmode == 'diff':
                cvmcep[:, 0] = 0.0
                wav = synthesizer.synthesis_diff(x,
                                                 cvmcep,
                                                 rmcep=mcep,
                                                 alpha=sconf.mcep_alpha,
                                                 )
                wavpath = os.path.join(test_dir, f + '_GAN_DIFFVC.wav')

            # write waveform
            wav = np.clip(wav, -32768, 32767)
            wavfile.write(wavpath, fs, wav.astype(np.int16))
            print(wavpath)


if __name__ == '__main__':
    main()
