# -*- coding: utf-8 -*-
"""
Created on 01/10/2018

@author: Danas Sutula

"""

import config

import os
import math
import time

import dolfin
import logging
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

from dolfin import *

import optim
import filters
import material
import utility

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():


    return


if __name__ == "__main__":

    plt.interactive(True)

    ### Plot solutions

    def plot(*args, **kwargs):
        '''Plot either `np.ndarray`s or something plottable from `dolfin`.'''
        plt.figure(kwargs.pop('name', None))
        if isinstance(args[0], (np.ndarray,list, tuple)):
            plt.plot(*args, **kwargs)
        else: # just try it anyway
            dolfin.plot(*args, **kwargs)
        plt.show()

    def plot_energy_vs_iterations():

        fh = plt.figure('energy_vs_iterations')
        fh.clear()

        plt.plot(hist_potential, '-.k')
        # plt.legend(['potential energy'])

        plt.ylabel('Strain energy')
        plt.xlabel('Iteration number')
        plt.title('Energy vs. Iterations')

        fh.tight_layout()
        plt.show()

    def plot_energy_vs_phasefields():

        EPS = 1e-12

        fh = plt.figure('energy_vs_phasefield')
        fh.clear()

        n = len(hist_phasefield)
        d = np.diff(hist_phasefield)

        ind = np.abs(d) > EPS
        ind = np.flatnonzero(ind)

        ind_first = (ind+1).tolist()
        ind_first.insert(0,0)

        ind_last = ind.tolist()
        ind_last.append(n-1)

        y_potential = np.array(hist_potential)[ind_last]
        x_phasefield = np.array(hist_phasefield)[ind_last]

        plt.plot(x_phasefield, y_potential, '-.b')

        plt.ylabel('Strain energy')
        plt.xlabel('Phasefield fraction')
        plt.title('Energy vs. Phasefield')

        fh.tight_layout()
        plt.show()

    def plot_phasefiled():

        fh = plt.figure('phasefield')
        fh.clear()

        dolfin.plot(p)

        plt.title('Phase-field, p\n(p_min = {0:.5}; p_max = {1:.5})'.format(
            p.vector().get_local().min(), p.vector().get_local().max()))

        fh.tight_layout()
        plt.show()

    plot_energy_vs_iterations()
    plot_energy_vs_phasefields()
    plot_phasefiled()

    plt.show()
