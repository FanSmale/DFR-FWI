# -*- coding: utf-8 -*-
"""
@Time: 2024/6/28

@author: Zeng Zifei
"""
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import matplotlib.ticker as ticker

mpl.use('TkAgg')
import scipy

font21 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
}

font18 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 35,
}


#
def pain_openfwi_seismic_data(para_seismic_data):
    """
    Plotting seismic dataset images of openfwi dataset

    :param para_seismic_data:   Seismic dataset (1000 x 70) (numpy)
    """
    data = cv2.resize(para_seismic_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)
    fig, ax = plt.subplots(figsize=(6.1, 8), dpi = 60)
    im = ax.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)

    ax.set_xticks(np.linspace(0, 0.7, 7))
    ax.set_yticks(np.linspace(0, 1.0, 9))
    ax.set_xticklabels(labels=[0, 0.11, 0.23, 0.35, 0.47, 0.59, 0.7], size=21)
    ax.set_yticklabels(labels=[0, 0.12, 0.25, 0.37, 0.5, 0.63, 0.75, 0.88, 1.0], size=21)
    # ax.set_xticks(np.linspace(0, 0.7, 5))
    # ax.set_yticks(np.linspace(0, 1.0, 5))
    # ax.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
    # ax.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.3)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    plt.show()
    plt.close()


def pain_seg_seismic_data(para_seismic_data):
    """
    Plotting seismic dataset images of SEG salt datasets

    :param para_seismic_data:  Seismic dataset (400 x 301) (numpy)
    :param is_colorbar: Whether to add a color bar (1 means add, 0 is the default, means don't add)
    """
    fig, ax = plt.subplots(figsize=(6.2, 8), dpi=120)

    im = ax.imshow(para_seismic_data, extent=[0, 300, 400, 0], cmap=plt.cm.seismic, vmin=-0.4, vmax=0.44)

    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)

    ax.set_xticks(np.linspace(0, 300, 5))
    ax.set_yticks(np.linspace(0, 400, 5))
    ax.set_xticklabels(labels=[0, 0.75, 1.5, 2.25, 3.0], size=21)
    ax.set_yticklabels(labels=[0.0, 0.50, 1.00, 1.50, 2.00], size=21)

    plt.rcParams['font.size'] = 14  # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.32)
    plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
    plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)

    plt.show()


def pain_openfwi_velocity_model(num, para_velocity_model, test_result_dir, min_velocity, max_velocity):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity model (70 x 70) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity model
    :param max_velocity:        Lower limit of velocity in the velocity model
    :return:
    '''

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(para_velocity_model, extent=[0, 0.7, 0.7, 0], vmin=min_velocity, vmax=max_velocity)

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=42)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=42)
    # 设置间距
    ax.tick_params(axis='x', which='major', pad=15)
    ax.tick_params(axis='y', which='major', pad=15)

    plt.rcParams['font.size'] = 46  # Set colorbar font size
    # 在ax的右侧创建一个坐标轴。cax的宽度为ax的3%，cax和ax之间的填充距固定为0.35英寸。
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.8)
    plt.colorbar(im, ax=ax, cax=cax, orientation='vertical',
                 ticks=np.linspace(min_velocity, max_velocity, 5), format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    plt.subplots_adjust(bottom=0.10, top=0.95, left=0.1, right=0.9)

    # plt.show()
    plt.savefig(test_result_dir + 'pd' + str(num))  # 设置保存名字
    plt.close('all')


def pain_openfwi_velocity_model2(num, output, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)
    im = ax.imshow(output, extent=[0, 0.7, 0.7, 0], vmin=vmin, vmax=vmax)
    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(np.linspace(0, 0.7, 8))
    ax.set_yticks(np.linspace(0, 0.7, 8))
    ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
    plt.rcParams['font.size'] = 14      # Set colorbar font size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.4)
    cb1 = plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal', format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
    tick_locator = ticker.MaxNLocator(nbins=9)
    cb1.locator = tick_locator
    cb1.set_ticks([np.min(vmin), 0.2*(vmax-vmin)+vmin, 0.4*(vmax-vmin)+vmin,
                   0.6*(vmax-vmin)+vmin, 0.8*(vmax-vmin)+vmin, np.max(vmax)])
    plt.subplots_adjust(bottom=0.11, top=0.97, left=0.12, right=0.97)
    plt.savefig(test_result_dir + 'PD' + str(num) + '.png')
    plt.close(fig)

def plot_openfwi_velocity_compare(num, output, target, test_result_dir, vmin, vmax):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    im = ax[0].matshow(output, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title('Prediction', y=1.1)
    ax[1].matshow(target, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title('Ground Truth', y=1.1)

    for axis in ax:
        # axis.set_xticks(range(0, 70, 10))
        # axis.set_xticklabels(range(0, 1050, 150))
        # axis.set_yticks(range(0, 70, 10))
        # axis.set_yticklabels(range(0, 1050, 150))

        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 700, 100))
        axis.set_yticks(range(0, 70, 10))
        axis.set_yticklabels(range(0, 700, 100))


        axis.set_ylabel('Depth (km)', fontsize=12)
        axis.set_xlabel('Position (km)', fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.75, label='Velocity(m/s)')
    plt.savefig(test_result_dir + 'PD' + str(num)) # 设置保存名字
    plt.close('all')


if __name__ == '__main__':
    # np.load('D:/Wang-Linrong/TFDSUNet-main /data/CurveVelA/train_data/seismic/seismic1.npy')
    # D:/Zhang-Xingyi/Data for FWI/SimulateData/train_data/georec_train/georec1.mat
    # D:/Zhang-Xingyi/DD-Net release/data/SEGSimulation/train_data/seismic/seismic1.mat

    # seismic = 'E:/Data/OpenFWI/CurveVelB/train_data/seismic/seismic1.npy'
    # pain_openfwi_seismic_data(np.load(seismic)[1, 2, :]) # 第一个数字是指哪一个地震数据，第二个数字是指第几炮
    seismic = 'E:/Data/seg/SEGSimulation/train_data/seismic/seismic1.mat'
    pain_seg_seismic_data(scipy.io.loadmat(seismic)["Rec"][:,:,15])  # 第三个数字是指第几炮 [400,301,29]