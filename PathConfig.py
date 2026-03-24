# -*- coding: utf-8 -*-
"""
@Time: 2025/10/29

@author: Zeng Zifei
"""
import os
from ParamConfig import *

###################################################
####                DATA   PATHS              #####
###################################################
DataSet = 'CurveVelB/'
Data_path = 'E:/Data/OpenFWI/' + DataSet


###################################################
####            RESULT   PATHS                #####
###################################################
main_dir = 'D:/PyCharm2024.3/PycharmProjects/DFR-FWI/'

# Check the main directory
if len(main_dir) == 0:
    raise Exception('Please specify path to correct directory!')

if os.path.exists('train_result/' + DataSet):
    train_result_dir = main_dir + 'train_result/' + DataSet  # Replace your dataset path here
    print(True)
else:
    os.makedirs('train_result/' + DataSet)
    train_result_dir = main_dir + 'train_result/' + DataSet
    print(False)

if os.path.exists('test_result/' + DataSet):
    test_result_dir = main_dir + 'test_result/' + DataSet  # Replace your dataset path here
else:
    os.makedirs('test_result/' + DataSet)
    test_result_dir = main_dir + 'test_result/' + DataSet

####################################################
####                   FileName                #####
####################################################
NoiseFlag = False  # If True add noise.
modelName = 'DFR_FWI'

tagM1 = '_TrainSize' + str(TrainSize)
tagM2 = '_Epoch' + str(Epochs)
tagM3 = '_BatchSize' + str(BatchSize)
tagM4 = '_LR' + str(LearnRate)

ModelName = modelName + tagM1 + tagM2 + tagM3 + tagM4

TestModelName = 'DDNet70'
PreModelname = TestModelName + '.pkl'

