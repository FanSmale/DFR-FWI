# -*- coding: utf-8 -*-
"""
@Time: 2024/6/28

@author: Zeng Zifei
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import scipy.io
from math import log10
from skimage.metrics import structural_similarity as ssim
from torch.autograd import Variable
from scipy.ndimage import uniform_filter
import pandas as pd
from PathConfig import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import lpips
lp = lpips.LPIPS(net='alex', version="0.1")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def model_reader(net, device, save_src='./models/SEGSimulation/model_name.pkl'):
    '''
    Read the .pkl model into the program

    :param net:         Variables used to store the .pkl model (need to open up the space in advance)
    :param device:      Equipment environment
    :param save_src:    The path where the model to be read is located
    :return:            Returns the variable "net", but points to what is read into the model
    '''

    print("The external .pkl model is about to be imported")
    print("Read file: {}".format(save_src))
    model = torch.load(save_src)
    try:                    # Attempt to do a network read
        net.load_state_dict(model)
    except RuntimeError:
        print("This model is obtained by multi-GPU training...")
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in model.items():
            name = k[7:]    # 7 is the length of the module
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)

    net = net.to(device)
    return net


def extract_contours(para_image):
    '''
    Use Canny to extract contour features

    :param image:       Velocity model (numpy)
    :return:            Binary contour structure of the velocity model (numpy)
    '''

    image = para_image
    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image_to_255 = norm_image * 255
    norm_image_to_255 = norm_image_to_255.astype(np.uint8)
    canny = cv2.Canny(norm_image_to_255, 10, 15)
    bool_canny = np.clip(canny, 0, 1)
    return bool_canny


def PSNR(target, prediction):
    psnr = 20 * log10(abs(target.max()) / np.sqrt(np.sum((target - prediction) ** 2) / prediction.size))
    return psnr


def SSIM_skimage(target, prediction):
    return ssim(target, prediction, data_range=target.max() - target.min(), multichannel=True)  #True（默认值），则假定图像为多通道图像，SSIM将在每个通道上计算并返回通道之间的平均值。


def MSE(prediction, target):
    prediction = Variable(torch.from_numpy(prediction))
    target = Variable(torch.from_numpy(target))
    criterion = nn.MSELoss(reduction='mean')
    MSE = criterion(prediction, target)
    return MSE.item()


def RMSE(prediction, target):
    prediction = Variable(torch.from_numpy(prediction))
    target = Variable(torch.from_numpy(target))
    criterion = nn.MSELoss(reduction='mean')
    MSE = criterion(prediction, target)
    RMSE = torch.sqrt(MSE)
    return RMSE.item()


def MAE(prediction, target):
    prediction = Variable(torch.from_numpy(prediction))
    target = Variable(torch.from_numpy(target))
    criterion = nn.L1Loss(reduction='mean')
    mae = criterion(prediction, target)
    return mae.item()


def _uqi_single(GT,P,ws):
    N = ws**2
    window = np.ones((ws,ws))

    GT_sq = GT*GT
    P_sq = P*P
    GT_P = GT*P

    GT_sum = uniform_filter(GT, ws)
    P_sum =  uniform_filter(P, ws)
    GT_sq_sum = uniform_filter(GT_sq, ws)
    P_sq_sum = uniform_filter(P_sq, ws)
    GT_P_sum = uniform_filter(GT_P, ws)

    GT_P_sum_mul = GT_sum*P_sum
    GT_P_sum_sq_sum_mul = GT_sum*GT_sum + P_sum*P_sum
    numerator = 4*(N*GT_P_sum - GT_P_sum_mul)*GT_P_sum_mul
    denominator1 = N*(GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
    denominator = denominator1*GT_P_sum_sq_sum_mul

    q_map = np.ones(denominator.shape)
    index = np.logical_and((denominator1 == 0) , (GT_P_sum_sq_sum_mul != 0))
    q_map[index] = 2*GT_P_sum_mul[index]/GT_P_sum_sq_sum_mul[index]
    index = (denominator != 0)
    q_map[index] = numerator[index]/denominator[index]

    s = int(np.round(ws/2))
    return np.mean(q_map[s:-s,s:-s])


def UIQ(GT,P,ws=8):
    if len(GT.shape) == 2:
        GT = GT[:, :, np.newaxis]
        P = P[:, :, np.newaxis]

    GT = GT.astype(np.float64)
    P = P.astype(np.float64)
    return np.mean([_uqi_single(GT[:,:,i],P[:,:,i],ws) for i in range(GT.shape[2])])


def LPIPS(GT, P):
    '''

    :param GT: numpy
    :param P: numpy
    :return:
    '''
    GT_tensor = torch.from_numpy(GT)
    P_tensor = torch.from_numpy(P)
    return lp.forward(GT_tensor, P_tensor).item()


def SaveTrainResults(train_loss, mae, mse, SavePath, ModelName, font2, font3):
    fig, ax = plt.subplots()

    plt.plot(train_loss[1:], label='Training')
    ax.set_xlabel('Number of epochs', font2)
    ax.set_ylabel('Loss', font2)
    ax.set_title('Training Loss', font3)

    ax.set_xlim([1, 10])
    ax.set_xticks([i for i in range(0, Epochs+1, 20)])
    ax.set_xticklabels((str(i) for i in range(0, Epochs+1, 20)))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)

    plt.savefig(SavePath + ModelName + 'TrainLoss.png', transparent=True)
    data = {'train_loss': train_loss, 'mae': mae, 'mse': mse}
    scipy.io.savemat(SavePath + ModelName + 'TrainLoss.mat', data)
    plt.show()
    plt.close()

def SaveTrainResults_3(train_loss, mae, mse, grad, SavePath, ModelName, font2, font3):
    fig, ax = plt.subplots()

    plt.plot(train_loss[1:], label='Training')
    ax.set_xlabel('Number of epochs', font2)
    ax.set_ylabel('Loss', font2)
    ax.set_title('Training Loss', font3)

    ax.set_xlim([1, 10])
    ax.set_xticks([i for i in range(0, Epochs+1, 20)])
    ax.set_xticklabels((str(i) for i in range(0, Epochs+1, 20)))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)

    plt.savefig(SavePath + ModelName + 'TrainLoss.png', transparent=True)
    data = {'train_loss': train_loss, 'mae': mae, 'mse': mse, 'grad': grad}
    scipy.io.savemat(SavePath + ModelName + 'TrainLoss.mat', data)
    plt.show()
    plt.close()

def SaveTrainResults_other(train_loss, pixel, ssim, grad, SavePath, ModelName, font2, font3):
    fig, ax = plt.subplots()

    plt.plot(train_loss[1:], label='Training')
    ax.set_xlabel('Number of epochs', font2)
    ax.set_ylabel('Loss', font2)
    ax.set_title('Training Loss', font3)

    ax.set_xlim([1, 10])
    ax.set_xticks([i for i in range(0, Epochs+1, 20)])
    ax.set_xticklabels((str(i) for i in range(0, Epochs+1, 20)))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)

    plt.savefig(SavePath + ModelName + 'TrainLoss.png', transparent=True)
    data = {'train_loss': train_loss, 'pixel': pixel, 'ssim': ssim, 'grad': grad}
    scipy.io.savemat(SavePath + ModelName + 'TrainLoss.mat', data)
    plt.show()
    plt.close()

def SaveTrainResults_2(train_loss, l1, logcosh, ssim, SavePath, ModelName, font2, font3):
    fig, ax = plt.subplots()

    plt.plot(train_loss[1:], label='Training')
    ax.set_xlabel('Number of epochs', font2)
    ax.set_ylabel('Loss', font2)
    ax.set_title('Training Loss', font3)

    ax.set_xlim([1, 10])
    ax.set_xticks([i for i in range(0, Epochs+1, 20)])
    ax.set_xticklabels((str(i) for i in range(0, Epochs+1, 20)))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)

    plt.savefig(SavePath + ModelName + 'TrainLoss.png', transparent=True)
    data = {'train_loss': train_loss, 'l1': l1, 'logcosh': logcosh, 'ssim': ssim}
    scipy.io.savemat(SavePath + ModelName + 'TrainLoss.mat', data)
    plt.show()
    plt.close()


def SaveTrainValidResults(train_loss, val_loss, l1, logcosh, SavePath, ModelName, font2, font3):
    fig, ax = plt.subplots()
    plt.plot(train_loss[1:], label='Training')
    plt.plot(val_loss[1:], label='Validation')
    ax.set_xlabel('Number of epochs', font2)
    ax.set_ylabel('Loss', font2)
    ax.set_title('Training and validation Loss', font3)

    ax.set_xlim([1, 10])
    ax.set_xticks([i for i in range(0, Epochs + 1, 20)])
    ax.set_xticklabels((str(i) for i in range(0, Epochs + 1, 20)))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)

    plt.savefig(SavePath + ModelName + 'TrainValidLoss.png', transparent=True)
    data = {'train_loss': train_loss, 'val_loss': val_loss, 'l1': l1, 'logcosh': logcosh}
    scipy.io.savemat(SavePath + ModelName + 'TrainValidLoss.mat', data)
    plt.show()
    plt.close()


def SaveTrainValidResults_2(train_loss, val_loss, l1, logcosh, ssim, SavePath, ModelName, font2, font3):
    fig, ax = plt.subplots()
    plt.plot(train_loss[1:], label='Training')
    plt.plot(val_loss[1:], label='Validation')
    ax.set_xlabel('Number of epochs', font2)
    ax.set_ylabel('Loss', font2)
    ax.set_title('Training and validation Loss', font3)

    ax.set_xlim([1, 10])
    ax.set_xticks([i for i in range(0, Epochs + 1, 20)])
    ax.set_xticklabels((str(i) for i in range(0, Epochs + 1, 20)))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)

    plt.savefig(SavePath + ModelName + 'TrainValidLoss.png', transparent=True)
    data = {'train_loss': train_loss, 'val_loss': val_loss, 'l1': l1, 'logcosh': logcosh, 'ssim': ssim}
    scipy.io.savemat(SavePath + ModelName + 'TrainValidLoss.mat', data)
    plt.show()
    plt.close()

def SaveTrainValidResults2(train_loss, val_loss, SavePath, ModelName, font2, font3):
    fig, ax = plt.subplots()
    plt.plot(train_loss[1:], label='Training')
    plt.plot(val_loss[1:], label='Validation')
    ax.set_xlabel('Number of epochs', font2)
    ax.set_ylabel('Loss', font2)
    ax.set_title('Training and validation Loss', font3)

    ax.set_xlim([1, 10])
    ax.set_xticks([i for i in range(0, Epochs + 1, 20)])
    ax.set_xticklabels((str(i) for i in range(0, Epochs + 1, 20)))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)

    plt.savefig(SavePath + ModelName + 'TrainValidLoss.png', transparent=True)
    data = {'train_loss': train_loss, 'val_loss': val_loss}
    scipy.io.savemat(SavePath + ModelName + 'TrainValidLoss.mat', data)
    plt.show()
    plt.close()


def SaveTrainValidGANResults(train_loss_g, train_loss_d, val_loss, SavePath, ModelName, font2, font3):
    fig, ax = plt.subplots()
    plt.plot(train_loss_g[1:], label='Training_g')
    plt.plot(train_loss_d[1:], label='Training_d')
    plt.plot(val_loss[1:], label='Validation')
    plt.legend()
    ax.set_xlabel('Num. of epochs', font2)
    ax.set_ylabel('Loss', font2)
    ax.set_title('Training and validation Loss', font3)
    ax.set_xlim([1, 10])
    ax.set_xticks([i for i in range(0, Epochs + 1, 20)])
    ax.set_xticklabels((str(i) for i in range(0, Epochs + 1, 20)))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
    ax.grid(linestyle='dashed', linewidth=0.5)

    plt.savefig(SavePath + ModelName + 'TrainLoss.png', transparent=True)
    data = {}
    data['train_loss_d'] = train_loss_g
    data['train_loss_d'] = train_loss_d
    data['val_loss'] = val_loss
    scipy.io.savemat(SavePath + ModelName + 'TrainValidLoss.mat', data)
    plt.show()
    plt.close()


def SaveTestResults(TotPSNR, TotSSIM, ToMSE, ToMAE, ToUQI, ToLPIPS, Prediction, GT, SavePath):
    data = {}
    data['TotPSNR'] = TotPSNR
    data['TotSSIM'] = TotSSIM
    data['ToMSE'] = ToMSE
    data['ToMAE'] = ToMAE
    data['ToUQI'] = ToUQI
    data['ToLPIPS'] = ToLPIPS
    data['GT'] = GT
    data['Prediction'] = Prediction
    print('TotPSNR: {}, TotSSIM: {},ToMSE: {}, ToMAE: {},ToUQI: {}, ToLPIPS: {}'.format(
        np.mean(TotPSNR), np.mean(TotSSIM), np.mean(ToMSE), np.mean(ToMAE), np.mean(ToUQI), np.mean(ToLPIPS)))

    file_path = SavePath + 'test_result.xlsx'
    df = pd.read_excel(file_path)

    Test_data = {
        'ModelName': TestModelName,
        'TotPSNR': np.mean(TotPSNR),
        'TotSSIM': np.mean(TotSSIM),
        'ToMSE': np.mean(ToMSE),
        'ToMAE': np.mean(ToMAE),
        'ToUQI': np.mean(ToUQI),
        'ToLPIPS': np.mean(ToLPIPS),
    }

    df = pd.concat([df, pd.DataFrame([Test_data])], ignore_index=True)
    df.to_excel(file_path, index=False)

    scipy.io.savemat(SavePath + TestModelName + '_TestResults.mat', data)


def SaveTestResults2(TotPSNR, TotSSIM, ToRMSE, ToMAE, ToUIQ, ToLPIPS, Prediction, GT, SavePath):
    data = {}
    data['TotPSNR'] = TotPSNR
    data['TotSSIM'] = TotSSIM
    data['ToRMSE'] = ToRMSE
    data['ToMAE'] = ToMAE
    data['ToUIQ'] = ToUIQ
    data['ToLPIPS'] = ToLPIPS
    data['GT'] = GT
    data['Prediction'] = Prediction
    print('TotPSNR: {}, TotSSIM: {},ToRMSE: {}, ToMAE: {},ToUIQ: {}, ToLPIPS: {}'.format(
        np.mean(TotPSNR), np.mean(TotSSIM), np.mean(ToRMSE), np.mean(ToMAE), np.mean(ToUIQ), np.mean(ToLPIPS)))

    file_path = SavePath + 'test_result2.xlsx'
    df = pd.read_excel(file_path)

    Test_data = {
        'ModelName': TestModelName,
        'TotPSNR': np.mean(TotPSNR),
        'TotSSIM': np.mean(TotSSIM),
        'ToRMSE': np.mean(ToRMSE),
        'ToMAE': np.mean(ToMAE),
        'ToUIQ': np.mean(ToUIQ),
        'ToLPIPS': np.mean(ToLPIPS),
    }

    df = pd.concat([df,pd.DataFrame([Test_data])],ignore_index=True)
    df.to_excel(file_path, index=False)

    scipy.io.savemat(SavePath + TestModelName + '_TestResults2.mat', data)


def SaveLearningRate(learning_rates, SavePath, ModelName):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, Epochs + 1), learning_rates, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs. Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(SavePath + ModelName + 'learning_rate_plot.png', transparent=True)  # 保存图像到指定目录
    learning_rates_dict = {'learning_rates': learning_rates}
    scipy.io.savemat(SavePath + ModelName + 'learning_rate.mat', learning_rates_dict)
    plt.show()

