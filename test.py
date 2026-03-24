# -*- coding: utf-8 -*-
"""
@Time : 2026/3/16 20:15

@Author : Zeng Zifei
"""
################################################
########        IMPORT LIBARIES         ########
################################################

import time
from func.data import *
from network.InversionNet import *
from network.DFR_FWI import *
from network.ABA_FWI import *
from network.DDNet70 import *

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

################################################
########         LOAD    NETWORK        ########
################################################

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

model_file = train_result_dir + PreModelname

if NetworkName == "DFR-FWI":
    net = DP_AttentionSkip_Biformer23()
elif NetworkName == "InversionNet":
    net = InversionNet()
elif NetworkName == "ABA-FWI":
    net = ABA_FWI()

model_param = torch.load(model_file, map_location=torch.device('cpu'))

net.load_state_dict(model_param)
################################################
########    LOADING TESTING DATA       ##【######
################################################

print('***************** Loading dataset *****************')

dataset_dir = Data_path

testSet = DatasetTestOpenFWI(dataset_dir, TestSize, 1, "test")  # 11 for test
test_loader = DataLoader(testSet, batch_size=TestBatchSize, shuffle=False)

################################################
########            TESTING             ########
################################################

print()
print('*******************************************')
print('*******************************************')
print('                  Testing...               ')
print('*******************************************')
print('*******************************************')
print()

since = time.time()

Total_PSNR = np.zeros((1, TestSize), dtype=float)
Total_SSIM = np.zeros((1, TestSize), dtype=float)
Total_RMSE = np.zeros((1, TestSize), dtype=float)
Total_MAE = np.zeros((1, TestSize), dtype=float)
Total_UIQ = np.zeros((1, TestSize), dtype=float)
Total_LPIPS = np.zeros((1, TestSize), dtype=float)

Prediction = np.zeros((TestSize, ModelDim[0], ModelDim[1]), dtype=float)
GT = np.zeros((TestSize, ModelDim[0], ModelDim[1]), dtype=float)
Prediction_N = np.zeros((3, ModelDim[0], ModelDim[1]), dtype=float)
GT_N = np.zeros((3, ModelDim[0], ModelDim[1]), dtype=float)

total = 0

for i, (seismic_datas, vmodels, vmodel_max_min) in enumerate(test_loader):
    net.eval()
    net.to(device)
    vmodels = vmodels[0].to(device)
    seismic_datas = seismic_datas[0].to(device)
    vmodel_max_min = vmodel_max_min[0].to(device)

    if NoiseFlag:
        seed = 42
        torch.manual_seed(seed)

        noise_mean = 0
        noise_std = 0.3
        noise = torch.normal(mean=noise_mean, std=noise_std, size=seismic_datas.shape).to(device)
        seismic_datas = seismic_datas + noise

    # Forward prediction
    outputs = net(seismic_datas)
    outputs = outputs.data.cpu().numpy()
    outputs = np.where(outputs > 0.0, outputs, 0.0)

    gts = vmodels.data.cpu().numpy()
    vmodel_max_min = vmodel_max_min.data.cpu().numpy()


    for k in range(outputs.shape[0]):
        pd = outputs[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        gt = gts[k, :, :, :].reshape(ModelDim[0], ModelDim[1])
        vmax = vmodel_max_min[k, 0]
        vmin = vmodel_max_min[k, 1]

        pd_N = pd * (vmax - vmin) + vmin
        gt_N = gt * (vmax - vmin) + vmin

        Prediction[i * TestBatchSize + k, :, :] = pd_N
        GT[i * TestBatchSize + k, :, :] = gt_N

        psnr = PSNR(gt, pd)
        ssim = SSIM_skimage(gt, pd)
        rmse = RMSE(pd, gt)
        mae = MAE(pd, gt)
        uiq = UIQ(pd, gt)
        lpips = LPIPS(pd, gt)


        Total_PSNR[0, total] = psnr
        Total_SSIM[0, total] = ssim
        Total_RMSE[0, total] = rmse
        Total_MAE[0, total] = mae
        Total_UIQ[0, total] = uiq
        Total_LPIPS[0, total] = lpips

        total = total + 1

        print('The %d testing psnr: %.2f, SSIM: %.4f, RMSE:  %.4f, MAE:  %.4f, UIQ:  %.4f, LPIPS: %.4f' % (total, psnr,
                                                                                                          ssim, rmse,
                                                                                                          mae, uiq,
                                                                                                          lpips))
time_elapsed = time.time() - since
print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
SaveTestResults2(Total_PSNR, Total_SSIM, Total_RMSE, Total_MAE, Total_UIQ, Total_LPIPS, Prediction, GT, test_result_dir)