from distutils.core import setup, Extension

module1 = Extension('spikingann',
                    sources = ['Simulation.c', "Brain.c", "Neuron.c"])

setup (name = 'spikingann',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])
