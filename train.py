# -*- coding: utf-8 -*-
"""
@Time : 2025/12/9 19:35

@Author : Zeng Zifei
"""

################################################
########            导入库               ########
################################################
import time
import datetime
from network.InversionNet import *
from network.DFR_FWI import *
from func.data import *
from func.utils import *
from func.loss import *
from torch.utils.tensorboard import SummaryWriter
from math import cos, pi
import os
from torch.cuda.amp import autocast, GradScaler


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

################################################
########             NETWORK            ########
################################################

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print(device)

if NetworkName == "DFR-FWI":
    net = DP_AttentionSkip_Biformer23()

elif NetworkName == "InversionNet":
    net = InversionNet()
elif NetworkName == "UNet_Biformer_DP_AttentionSkip":
    net = DP_AttentionSkip_Biformer23()

net = net.to(device)

# Optimizer we want to use
optimizer = torch.optim.Adam(net.parameters(), lr=LearnRate)


def warmup_cosine(optimizer, current_epoch, max_epoch, lr_min=0.0001, lr_max=0.001, warmup_epoch = 10):
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

lr_max = 0.001
lr_min = 0.0005
warmup_epoch = 15

# If ReUse, it will load saved model from premodelfilepath and continue to train
if ReUse:
    print('***************** Loading pre-training model *****************')
    print('')
    premodel_file = train_result_dir + PreModelname
    net.load_state_dict(torch.load(premodel_file))
    net = net.to(device)
    print('Finish downloading:', str(premodel_file))

################################################
########    LOADING TRAINING DATA       ########
################################################
print('***************** Loading training dataset *****************')

dataset_dir = Data_path
trainSet = DatasetOpenFWI(dataset_dir, TrainSize, 1, "train")
train_loader = DataLoader(trainSet, batch_size=BatchSize, shuffle=True)


################################################
########            TRAINING            ########
################################################

# 创建TensorBoard writer
writer = SummaryWriter(log_dir="tensorboard_logs/cvb/DFR_FWI")

print()
print('*******************************************')
print('*******************************************')
print('                Training ...               ')
print('*******************************************')
print('*******************************************')
print()

print('网络:%s' % str(NetworkName))
print('原始地震数据尺寸:%s' % str(DataDim))
print('原始速度模型尺寸:%s' % str(ModelDim))
print('训练规模:%d' % int(TrainSize))
print('训练批次大小:%d' % int(BatchSize))
print('迭代轮数:%d' % int(Epochs))
print('学习率:%.5f' % float(LearnRate))
print('优化器: adam')
print('学习率调度: warmup+余弦退火')
print('loss采用的l1+l2+grad_loss')

start = time.time()
learning_rates = []

def train():
    total_loss = 0
    total_loss_mae = 0
    total_loss_mse = 0
    total_loss_grad = 0

    for i, (seismic_datas, vmodels) in enumerate(train_loader):
        net.train()

        seismic_datas = seismic_datas[0].to(device)
        vmodels = vmodels[0].to(device).to(torch.float32)

        optimizer.zero_grad()

        if NoiseFlag:
            noise_mean = 0
            noise_std = 0.2
            noise = torch.normal(mean=noise_mean, std=noise_std, size=seismic_datas.shape).to(device)
            seismic_datas = seismic_datas + noise

        outputs = net(seismic_datas)
        outputs = outputs.to(torch.float32)
        vmodels = vmodels.to(torch.float32)

        loss, loss_mae, loss_mse, loss_grad = criterion_pixel_grad(outputs, vmodels)

        if np.isnan(float(loss_mae.item())):
            raise ValueError('pixel loss is nan while training')
        if np.isnan(float(loss_mse.item())):
            raise ValueError('ssim loss is nan while training')
        if np.isnan(float(loss_grad.item())):
            raise ValueError('grad loss is nan while training')
        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        total_loss += loss.item()
        total_loss_mae += loss_mae.item()
        total_loss_mse += loss_mse.item()
        total_loss_grad += loss_grad.item()

        loss = loss.to(torch.float32)
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    avg_loss_mae = total_loss_mae / len(train_loader)
    avg_loss_mse = total_loss_mse / len(train_loader)
    avg_loss_grad = total_loss_grad / len(train_loader)

    return avg_loss, avg_loss_mae, avg_loss_mse, avg_loss_grad
    # return avg_loss


train_loss_list = 0
train_loss_mae_list = 0
train_loss_mse_list = 0
train_loss_grad_list = 0

for epoch in range(Epochs):
    epoch_loss = 0.0
    since = time.time()

    warmup_cosine(optimizer=optimizer, current_epoch=epoch+1, max_epoch=Epochs, lr_min=lr_min, lr_max=lr_max, warmup_epoch=warmup_epoch)

    train_loss, train_loss_mae, train_loss_mse, train_loss_grad = train()

    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    print(optimizer.param_groups[0]['lr'])

    if (epoch % 1) == 0:
        print(f"Epoch: {epoch + 1}, Train loss:{train_loss:.4f}")
        print(f"mae: {train_loss_mae: .6f}, mse:{train_loss_mse: .6f}, grad:{train_loss_grad: .10f}")
        time_elapsed = time.time() - since
        print('Epoch consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if (epoch + 1) % SaveEpoch == 0:
        torch.save(net.state_dict(), train_result_dir + ModelName + '_epoch' + str(epoch + 1) + '.pkl')
        print('Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))

    train_loss_list = np.append(train_loss_list, train_loss)
    train_loss_mae_list = np.append(train_loss_mae_list, train_loss_mae)
    train_loss_mse_list = np.append(train_loss_mse_list, train_loss_mse)
    train_loss_grad_list = np.append(train_loss_grad_list, train_loss_grad)

# Record the consuming time
time_elapsed = time.time() - start
print('Training complete in {:.0f}m  {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# Save the loss
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 21,
         }

SaveTrainResults_3(train_loss=train_loss_list, mae=train_loss_mae_list, mse=train_loss_mse_list, grad=train_loss_grad_list, SavePath=train_result_dir, ModelName=ModelName, font2=font2, font3=font3)
SaveLearningRate(learning_rates=learning_rates, SavePath=train_result_dir, ModelName=ModelName)



