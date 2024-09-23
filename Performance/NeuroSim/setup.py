import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

ext_modules = [
    Extension(
    'neurosim',
        ['Adder.cpp'          ,'Chip.cpp'             ,'DFF.cpp'                    ,'global.cpp'           ,'MultilevelSAEncoder.cpp',
        'ProcessingUnit.cpp'  ,'SenseAmp.cpp'         ,'SubArray.cpp'               ,'VoltageSenseAmp.cpp'  ,'AdderTree.cpp' ,'Comparator.cpp',
        'DRAM.cpp'            ,'HTree.cpp'            ,'MultilevelSenseAmp.cpp'     ,'NewMux.cpp'           ,'ShiftAdd.cpp',   
        'SwitchMatrix.cpp'    ,'WLDecoderOutput.cpp'  ,'BitShifter.cpp'             ,'CurrentSenseAmp.cpp'  ,'formula.cpp' ,'LevelShifter.cpp',
        'NewSwitchMatrix.cpp' ,'ReadCircuit.cpp'      ,'Sigmoid.cpp'                ,'Technology.cpp'       ,'WLNewDecoderDriver.cpp' ,'Buffer.cpp' ,'DecoderDriver.cpp',
        'funcs.cpp'           ,'Mux.cpp'              ,'Param.cpp'                  ,'RowDecoder.cpp'       ,'SramNewSA.cpp',
        'Bus.cpp'             ,'DeMux.cpp'            ,'FunctionUnit.cpp'           ,'MaxPooling.cpp'       ,'neurosim.cpp','Precharger.cpp',
        'SarADC.cpp'          ,'SRAMWriteDriver.cpp'  ,'Tile.cpp'                   ,'RecfgAdderTree.cpp'   ,'Multiplier.cpp'
        ],
        include_dirs=['/home/wangcong/anaconda3/lib/python3.11/site-packages/pybind11/include'],
        # use the  pybind11 path in your computer
        
    language='c++',
    ),
]

setup(
    name='neurosim',
    version='1.3',
    author='Cong WANG',
    author_email='cwang841@connect.hkust-gz.edu.cn',
    description='python wrapper for neurosim 1.3',
    ext_modules=ext_modules,
)