'''Configure printing, plotting, logging options.'''

import numpy
numpy.set_printoptions(
    edgeitems = 5,
    threshold = 100,
    formatter = {'float' : '{: 13.6e}'.format},
    linewidth = 160)

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.interactive(True)

import logging
logging.basicConfig(
    level=logging.WARNING,
    format='[%(levelname)s:%(funcName)s] %(message)s')

import dolfin
dolfin.set_log_level(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)
