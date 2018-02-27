from distutils.core import setup, Extension
import numpy

module1 = Extension('spikingann',
    sources = ['SpikingANN.c', "Brain.c", "Neuron.c"],
    include_dirs=[numpy.get_include()],
)

setup (name = 'spikingann',
       version = '0.1.2',
       description = 'This is a library for using a spiking artificial neural network',
       ext_modules = [module1])
