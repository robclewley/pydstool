#!/sw/bin/python2.3
#Author:    Drew LaMar
#Date:      1 April 2006
#$Revision: 0.2.2 $

from distutils.core import setup, Extension
import os

setup(name='auto',
      version='1.0',
      description='AUTO module created automatically by PyDSTool',
      include_dirs=['src/include', 'module/include'],
      ext_modules=[Extension('_auto',
      	sources=['module/automod.c', 'module/interface.c', 'module/automod.i'],
	extra_compile_args=['-w','-DPYTHON_MODULE'], 
        library_dirs=['lib'],
        libraries=['auto2000']
)])

os.system('mv build/lib.darwin-8.6.0-PowerMacintosh-2.3/_auto.so module')
os.system('rm -rf build')
