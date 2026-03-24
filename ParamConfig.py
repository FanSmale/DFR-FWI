# -*- coding: utf-8 -*-
"""
@Time: 2025/10/29

@author: Zeng Zifei
"""

####################################################
####             MAIN PARAMETERS                ####
####################################################
DataSet = 'CurveVelB/'
NetworkName = "DFR-FWI"

# OpenFWI = True
# Marmousi = False
# SEGSaltData = False
# SEGSimulation = False

ReUse = False  # If False always re-train a network

dh = 10  # Space interval

####################################################
####             NETWORK PARAMETERS             ####
####################################################

if NetworkName in ["InversionNet", "VelocityGAN", "ABA-FWI"]:
    LearnRate = 1e-4
elif NetworkName == "DFR-FWI":
    LearnRate = 1e-3
else:
    LearnRate = 1e-5


if DataSet == "FlatVelA/":
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    OutChannel = 1
    Epochs = 120
    TrainSize = 24000
    ValSize = 500
    TestSize = 6000
    TestBatchSize = 20
    BatchSize = 20
    SaveEpoch = 10
elif DataSet == "FlatFaultA/":
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    OutChannel = 1
    Epochs = 400
    TrainSize = 48000
    ValSize = 500
    TestSize = 100
    TestBatchSize = 20
    BatchSize = 20
    SaveEpoch = 10
elif DataSet == "CurveVelA/":
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    OutChannel = 1
    Epochs = 160
    TrainSize = 24000
    ValSize = 1000
    TestSize = 6000
    TestBatchSize = 20
    BatchSize = 20
    SaveEpoch = 10
elif DataSet == "CurveFaultA/":
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    OutChannel = 1
    Epochs = 160
    TrainSize = 48000
    ValSize = 500
    TestSize = 6000
    TestBatchSize = 20
    BatchSize = 20
    SaveEpoch = 10

elif DataSet == "CurveVelB/":
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    OutChannel = 1
    Epochs = 160
    TrainSize = 24000
    ValSize = 500
    TestSize = 6000
    TestBatchSize = 20
    BatchSize = 20
    SaveEpoch = 10

elif DataSet == "CurveFaultB/":
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    OutChannel = 1
    Epochs = 200
    TrainSize = 48000
    ValSize = 500
    TestSize = 6000
    TestBatchSize = 20
    BatchSize = 20
    SaveEpoch = 10

elif DataSet == "SEGSimulation/":
    DataDim = [400, 301]
    ModelDim = [201, 301]
    InChannel = 29
    OutChannel = 1
    Epochs = 200
    TrainSize = 1600
    ValSize = 20
    TestSize = 100
    TestBatchSize = 10
    BatchSize = 4
    SaveEpoch = 10
elif DataSet == "SEGSaltData/":
    DataDim = [400, 301]
    ModelDim = [201, 301]
    InChannel = 29
    OutChannel = 1
    Epochs = 120
    TrainSize = 130
    ValSize = 5
    TestSize = 10
    TestBatchSize = 5
    BatchSize = 10
    SaveEpoch = 10
elif DataSet == 'marmousi_70_70/':
    DataDim = [1000, 70]
    ModelDim = [70, 70]
    InChannel = 5
    Epochs = 400
    TrainSize = 30926
    TestSize = 328
    ValSize = 100
    BatchSize = 64
    TestBatchSize = 8
    SaveEpoch = 10

else:
    print('The selected dataset is invalid')
    exit(0)
