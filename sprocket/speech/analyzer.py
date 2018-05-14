# -*- coding: utf-8 -*-

import pyworld
import numpy as np

class WORLD(object):
    """WORLD-based speech analyzer

    Parameters
    ----------
    fs : int, optional
        Sampling frequency
        Default set to 16000
    fftl : int, optional
        FFT length
        Default set to 1024
    shiftms : int, optional
        Shift lengs [ms]
        Default set to 5.0
    minf0 : int, optional
        Floor in f0 estimation
        Default set to 50
    maxf0 : int, optional
        Ceil in f0 estimation
        Default set to 500
    """

    def __init__(self, fs=16000, fftl=1024, shiftms=5.0, minf0=40.0, maxf0=500.0):
        self.fs = fs
        self.fftl = fftl
        self.shiftms = shiftms
        self.minf0 = minf0
        self.maxf0 = maxf0

    def cal_rmse(self, pre, tar):
        return np.sqrt(((pre - tar) ** 2).mean())

    def analyze(self, x, in_f0=1):
        """Analyze acoustic features based on WORLD

        analyze F0, spectral envelope, aperiodicity

        Paramters
        ---------
        x : array, shape (`T`)
            monoral speech signal in time domain

        Returns
        ---------
        f0 : array, shape (`T`,)
            F0 sequence
        spc : array, shape (`T`, `fftl / 2 + 1`)
            Spectral envelope sequence
        ap: array, shape (`T`, `fftl / 2 + 1`)
            aperiodicity sequence

        """
        f0, time_axis = pyworld.harvest(x, self.fs, f0_floor=self.minf0,
                                        f0_ceil=self.maxf0, frame_period=self.shiftms)
        f0_tmp = f0
        if in_f0 is not None:
            print("exchange f0")
            assert len(f0) == len(in_f0)
            f0 = in_f0
        spc = pyworld.cheaptrick(x, f0, time_axis, self.fs,
                                 fft_size=self.fftl)
        spc_tmp = pyworld.cheaptrick(x, f0_tmp, time_axis, self.fs,
                                 fft_size=self.fftl)
        ap = pyworld.d4c(x, f0, time_axis, self.fs, fft_size=self.fftl)
        ap_tmp = pyworld.d4c(x, f0_tmp, time_axis, self.fs, fft_size=self.fftl)

        # for check RMSE
        rmse_f0 = self.cal_rmse(f0_tmp, f0)
        print('f0 RMSE : ', rmse_f0)
        rmse_spc = self.cal_rmse(spc_tmp, spc)
        print('spc RMSE : ', rmse_spc)
        rmse_ap = self.cal_rmse(ap_tmp, ap)
        print('ap RMSE : ', rmse_ap)

        assert spc.shape == ap.shape
        return f0, spc, ap

    def analyze_f0(self, x):
        """Analyze decomposes a speech signal into F0:

        Paramters
        ---------
        x: array, shape (`T`)
            monoral speech signal in time domain

        Returns
        ---------
        f0 : array, shape (`T`,)
            F0 sequence

        """

        f0, time_axis = pyworld.harvest(x, self.fs, f0_floor=self.minf0,
                                        f0_ceil=self.maxf0, frame_period=self.shiftms)

        return f0

    def synthesis(self, f0, spc, ap):
        """Synthesis re-synthesizes a speech waveform from:

        Parameters
        ----------
        f0 : array, shape (`T`)
            F0 sequence
        spc : array, shape (`T`, `dim`)
            Spectral envelope sequence
        ap: array, shape (`T`, `dim`)
            Aperiodicity sequence

        """

        return pyworld.synthesize(f0, spc, ap, self.fs, frame_period=self.shiftms)
