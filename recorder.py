import matplotlib
import pandas as pd
import cmaps
from global_land_mask import globe
from scipy import interpolate
import cartopy.mpl.ticker as cticker
import xarray as xr
import os
import imageio
import cartopy.crs as ccrs
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import json
from torch.optim.lr_scheduler import ExponentialLR
import scipy.stats as sps
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False
c1 = cmaps.temp_19lev
matplotlib.use('TkAgg')


# plt.rcParams['font.sans-serif'] = ['SimHei'] #中文支持
# %matplotlib inline
class Recorder:
    def __init__(self, verbose=False, delta=0):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving models ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoints.pth')
        self.val_loss_min = val_loss


nino34 = [25, 35, 100, 150]
nino3 = [25, 35, 120, 180]
nino4 = [25, 35, 70, 120]
ido_west = [10, 14, 10, 14]
ido_east = [10, 12, 18, 22]


nino_dict = {0: nino34, 1: nino3, 2: nino4, 3: ido_west, 4: ido_east}


class Pred_True_vision:

    def __init__(self, pred, true, save_plt_path):
        self.pred = pred  # B,T,H,W
        self.true = true
        self.nino34_region = nino_dict[0]
        self.nino3_region = nino_dict[1]
        self.nino4_region = nino_dict[2]
        self.ido_west_region = nino_dict[3]
        self.ido_east_region = nino_dict[4]
        self.save_plt_path = save_plt_path
        self.acc_weight = np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6+[5]) * np.log(np.arange(25) + 1)

    def score(self, y_pred, y_true):
        pred = y_pred - np.mean(y_pred, axis=0, keepdims=True)

        true = y_true - np.mean(y_true, axis=0, keepdims=True)  # (N, 24)
        cor = (pred * true).sum(axis=0) / (np.sqrt(np.sum(pred ** 2, axis=0) * np.sum(true ** 2, axis=0)) + 1e-6)
        acc = (self.acc_weight[0:cor.shape[0]] * cor).sum()
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0)).sum()
        return 2 / 3. * acc - rmse, rmse

    def pers_nino_region(self, nino_region, score_acu):
        ACCList = []
        PList = []
        nino_region_one = self.nino34_region
        ido_region_west = 0
        ido_region_east = 0
        if nino_region == 'nino34':
            print('nino34 region')
        elif nino_region == 'nino3':
            nino_region_one = self.nino3_region
            print('nino3 region')
        elif nino_region == 'nino4':
            nino_region_one = self.nino4_region
            print('nino4 region')
        else:
            ido_region_west = self.ido_west_region
            ido_region_east = self.ido_east_region
            print('ido region')

        if not os.path.exists(self.save_plt_path):
            os.makedirs(self.save_plt_path)

        pred = self.pred
        true = self.true
        # pred = pred.reshape(T,V)
        if 'nino' in nino_region:
            pred_array = np.mean(
                pred[:, :,  nino_region_one[0]:nino_region_one[1] + 1, nino_region_one[2]:nino_region_one[3] + 1],
                axis=(2, 3))
            true_array = np.mean(
                true[:, :,  nino_region_one[0]:nino_region_one[1] + 1, nino_region_one[2]:nino_region_one[3] + 1],
                axis=(2, 3))

            pred_list = np.swapaxes(pred_array, 0, 1)

            true_list = np.swapaxes(true_array, 0, 1)
        else:
            pred_array_west = np.mean(
                pred[:, :,  ido_region_west[0]:ido_region_west[1] + 1, ido_region_west[2]:ido_region_west[3] + 1],
                axis=(2, 3))
            true_array_west = np.mean(
                true[:, :, ido_region_west[0]:ido_region_west[1] + 1, ido_region_west[2]:ido_region_west[3] + 1],
                axis=(2, 3))

            pred_array_east = np.mean(
                pred[:, :,  ido_region_east[0]:ido_region_east[1] + 1, ido_region_east[2]:ido_region_east[3] + 1],
                axis=(2, 3))
            true_array_east = np.mean(
                true[:, :, ido_region_east[0]:ido_region_east[1] + 1, ido_region_east[2]:ido_region_east[3] + 1],
                axis=(2, 3))

            pred_array = pred_array_west - pred_array_east
            true_array = true_array_west - true_array_east

            pred_list = np.swapaxes(pred_array, 0, 1)

            true_list = np.swapaxes(true_array, 0, 1)

        t, _ = pred_list.shape

        # 对t个月计算相关系数
        for index_month in range(t):
            acc, p_value = sps.pearsonr(pred_list[index_month], true_list[index_month])
            ACCList.append(acc)
            PList.append(p_value)
        if score_acu == True:
            score, nino34_rmse = self.score(pred_array, true_array)
            return score, nino34_rmse, ACCList, PList
        else:
            return ACCList, PList

    def trainPlot(self, ACCList, iter, save_root, model_name, nino_region):

        fig = plt.figure(figsize=(10, 9))
        ax1 = fig.add_subplot(211)
        len1 = len(ACCList)
        avg_score = sum(ACCList) / len1
        ax1.plot(np.arange(1, len1 + 1), ACCList, "-o", label=f"{model_name}-{avg_score:.3f}")

        # # 在每个点旁边显示数值
        for i in range(len(ACCList)):
            ax1.text(i + 1, ACCList[i], f'{ACCList[i]:.3f}', ha='center', va='bottom')  # ha和va控制文本的水平和垂直对齐方式

        ax1.hlines(0.5, 0.5, len1 + 1 + 0.5)
        ax1.set_xlim(0.5, len1 + 1 + 0.5)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel("lead time (month)-{}".format(nino_region))
        ax1.set_ylabel("ACC")
        plt.legend()

        save_path = os.path.join(save_root, f"{nino_region}")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(fr"{save_path}/corr_{nino_region}_{iter}.png", dpi=100)
        # plt.show()
        plt.close()


class test_vision_exp:
    def __init__(self, vision_time, save_path):

        self.vision_time = vision_time
        self.save_path = save_path

    def reback_show(self, data):
        # # Set a latitude and longitude coordinate grid with an interval of 1°, and use the interpolation function to obtain the SST values at the grid points of this coordinate grid.

        lat = np.arange(-59.5, 59.5, 5, )
        lon = np.arange(0, 360, 5, )
        xx, yy = np.meshgrid(lon, lat)
        data_one = data.data
        z = data_one
        f = interpolate.interp2d(lon, lat, np.array(z), kind='cubic')

        xnew = np.arange(0, 360, 1, )
        ynew = np.arange(-59.5, 60.5, 1, )

        znew = f(xnew, ynew)

        lon_grid, lat_grid = np.meshgrid(xnew - 180, ynew)
        is_on_land = globe.is_land(lat_grid, lon_grid)
        is_on_land = np.concatenate([is_on_land[:, xnew >= 180], is_on_land[:, xnew < 180]], axis=1)

        znew[np.squeeze(is_on_land)] = np.nan
        return np.flipud(znew)

    def vision_plot(self, pred, true, vision_time, iter):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1行2列的子图
        # 左侧子图
        ax1.imshow(pred, cmap=plt.cm.RdBu_r)
        ax1.set_title('Pre-SST')
        ax1.set_xlabel('lon')
        ax1.set_ylabel('lat')

        # 右侧子图
        ax2.imshow(true, cmap=plt.cm.RdBu_r)
        ax2.set_title('True-SST')
        ax2.set_xlabel('lon')
        ax2.set_ylabel('lat')
        # 调整布局
        plt.tight_layout()

        # 显示图形
        # plt.show()

        # save plt
        plt.savefig(fr"{self.save_path}/{iter}.png", dpi=100)
        plt.close()

    def visualize_u_v_sst(self, data, one_time):
        print('data', data.shape)
        u = self.reback_show(data[0])
        v = self.reback_show(data[1])
        sst = self.reback_show(data[2])

        w = np.sqrt(u * u + v * v)
        xnew = np.arange(0, 360, 1, )
        ynew = np.arange(-59.5, 60.5, 1, )
        lon = xnew
        lat = ynew
        time = one_time

        def make_map(ax, title, box, xstep, ystep):
            # set_extent  set crs
            ax.set_extent(box, crs=ccrs.PlateCarree())
            ax.coastlines(scale)  # set coastline resolution
            # set coordinate axis
            ax.set_xticks(np.arange(box[0], box[1], xstep)[1:], crs=ccrs.PlateCarree())
            ax.set_yticks(np.array([-45, -30, -15, 0, 15, 30, 45]), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
            # 经度0不加标识
            ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
            ax.set_title(title, fontsize=15, loc='center')
            return ax

        fig = plt.figure(figsize=(12, 8), dpi=100)
        x, y = np.meshgrid(lon, lat)
        # print(x)
        # print(y)

        # lat = np.arange(-59.5, 59.5, 5, )
        # lon = np.arange(0, 360, 5, )
        # 控制画图的边界大小
        box1 = [0, 359, -59.5, 59.5]
        scale = '50m'

        proj = ccrs.PlateCarree(central_longitude=180)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        make_map(ax, str(time[0]), box1, 30, 10)

        cb = ax.quiver(x[::5, ::5], y[::5, ::5], u[::-5, ::5], v[::-5, ::5], pivot='mid',
                       width=0.0028, scale=150, transform=ccrs.PlateCarree(), color='k', angles='xy', zorder=1)

        cp = ax.contourf(lon, lat, sst[::-1, :], zorder=0, transform=ccrs.PlateCarree(),
                         cmap=plt.cm.RdBu_r, extend='both', levels=np.linspace(-3, 3, 40))

        cbar = fig.colorbar(mappable=cp, orientation='horizontal', extend='both', pad=0.06, shrink=0.5)


        ax.set_title('u-v-sst-{}'.format(time))
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        # 调整布局
        plt.tight_layout()
        plt.show()

    def make_git(self):
        folder_path = self.save_path
        output_path = f'{self.save_path}/output.gif'

        images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                  file.endswith(('.png', '.jpg', '.jpeg'))]

        images.sort()

        # 读取图片并保存为GIF
        with imageio.get_writer(output_path, mode='I', duration=500) as writer:
            for image_file in images:
                image = imageio.imread(image_file)
                writer.append_data(image)

        print(f'GIF has been saved to {output_path}')

    def make_plt(self, pred, true):
        t, v, h, w = pred.shape
        for i in range(t):
            pre_back = self.reback_show(pred[i, 2])
            true_back = self.reback_show(true[i, 2])
            self.vision_plot(pre_back, true_back, self.vision_time[i], i)

