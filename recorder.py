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
nino3 = [25, 35, 120, 170]
nino4 = [25, 35, 70, 120]
ido_west = [10, 14, 10, 14]
ido_east = [10, 12, 18, 22]
# IOD指数是通过计算印度洋西部（50°E～70°E, 10°S～10°N）与东部（90°E～110°E, 10°S～0°）区域平均海表温度距平之差来定义的。当IOD指数大于0.4时


nino_dict = {0: nino34, 1: nino3, 2: nino4, 3: ido_west, 4: ido_east}


class Pred_True_vision:

    def __init__(self, pred, true, save_plt_path):
        self.pred = pred  # B,T,H,W
        self.true = true
        # self.lat=np.array([-59.5, -54.5, -49.5, -44.5, -39.5, -34.5, -29.5, -24.5, -19.5, -14.5, -9.5, -4.5, 0.5, 5.5, 10.5, 15.5, 20.5, 25.5, 30.5, 35.5, 40.5, 45.5, 50.5, 55.5])
        # self.lon=np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355])
        self.nino34_region = nino_dict[0]
        self.nino3_region = nino_dict[1]
        self.nino4_region = nino_dict[2]
        self.ido_west_region = nino_dict[3]
        self.ido_east_region = nino_dict[4]
        self.save_plt_path = save_plt_path
        self.acc_weight = np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6+[5]) * np.log(np.arange(25) + 1)

    def score(self, y_pred, y_true):
        # for pytorch
        pred = y_pred - np.mean(y_pred, axis=0, keepdims=True)

        true = y_true - np.mean(y_true, axis=0, keepdims=True)  # (N, 24)
        cor = (pred * true).sum(axis=0) / (np.sqrt(np.sum(pred ** 2, axis=0) * np.sum(true ** 2, axis=0)) + 1e-6)
        print(cor.shape[0])
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
        B, T, V, H, W = pred.shape
        # pred = pred.reshape(T,V)
        if 'nino' in nino_region:
            pred_array = np.mean(
                pred[:, :, 2, nino_region_one[0]:nino_region_one[1] + 1, nino_region_one[2]:nino_region_one[3] + 1],
                axis=(2, 3))
            true_array = np.mean(
                true[:, :, 2, nino_region_one[0]:nino_region_one[1] + 1, nino_region_one[2]:nino_region_one[3] + 1],
                axis=(2, 3))

            pred_list = np.swapaxes(pred_array, 0, 1)

            true_list = np.swapaxes(true_array, 0, 1)
        else:
            pred_array_west = np.mean(
                pred[:, :, 2, ido_region_west[0]:ido_region_west[1] + 1, ido_region_west[2]:ido_region_west[3] + 1],
                axis=(2, 3))
            true_array_west = np.mean(
                true[:, :, 2, ido_region_west[0]:ido_region_west[1] + 1, ido_region_west[2]:ido_region_west[3] + 1],
                axis=(2, 3))

            pred_array_east = np.mean(
                pred[:, :, 2, ido_region_east[0]:ido_region_east[1] + 1, ido_region_east[2]:ido_region_east[3] + 1],
                axis=(2, 3))
            true_array_east = np.mean(
                true[:, :, 2, ido_region_east[0]:ido_region_east[1] + 1, ido_region_east[2]:ido_region_east[3] + 1],
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
        # # 设置间隔为1°的经纬度坐标网格，用插值函数得到该坐标网格点的SST值

        lat = np.arange(-59.5, 59.5, 5, )
        lon = np.arange(0, 360, 5, )
        # 以纬度和经度生成网格点坐标矩阵
        xx, yy = np.meshgrid(lon, lat)
        data_one = data.data
        # 取样本0第0月的SST值
        z = data_one
        # 采用三次多项式插值，得到z = f(x, y)的函数f
        f = interpolate.interp2d(lon, lat, np.array(z), kind='cubic')

        xnew = np.arange(0, 360, 1, )
        ynew = np.arange(-59.5, 60.5, 1, )

        znew = f(xnew, ynew)
        # 判断坐标矩阵上的网格点是否为陆地

        lon_grid, lat_grid = np.meshgrid(xnew - 180, ynew)
        is_on_land = globe.is_land(lat_grid, lon_grid)
        is_on_land = np.concatenate([is_on_land[:, xnew >= 180], is_on_land[:, xnew < 180]], axis=1)

        # 同样进行陆地掩膜
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
        print(sst.shape)
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
        # 调整 colorbar 的大小和间距
        # cbar.ax.tick_params(labelsize=10)  # 调整刻度标签的大小
        # cbar.fraction = 0.0046  # 控制 colorbar 的大小
        # cbar.pad = 0.004  # 控制 colorbar 和图形之间的间距

        ax.set_title('u-v-sst-{}'.format(time))
        ax.set_xlabel('lon')
        ax.set_ylabel('lat')
        # 调整布局
        plt.tight_layout()
        plt.show()

    def make_git(self):
        # 设置图片文件夹路径和GIF输出路径
        folder_path = self.save_path
        output_path = f'{self.save_path}/output.gif'

        # 获取文件夹中所有图片的路径
        images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                  file.endswith(('.png', '.jpg', '.jpeg'))]

        # 对图片进行排序，确保它们按照正确的顺序
        images.sort()  # 你可以根据需要添加更复杂的排序逻辑

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


if __name__ == '__main__':
    # vision_time=pd.date_range(start='2015-01-01',end='2016-12-01',freq='MS')
    # date_list = vision_time.strftime('%Y-%m-%d').tolist()
    # # print(date_list)
    path = r'F:\ENSO-Data\code-2\models\smivp\test'
    #
    # pred=xr.open_zarr(r'F:\ENSO-Data\GODAS\godas_min.zarr',consolidated=True).sel(time=slice('2015-01-01','2016-12-01'))['godas_data'].fillna(0).values
    # print(pred.shape)
    # xx=np.reshape(pred,(1,24,12,24,72))
    # pred_exp=np.concatenate((xx,xx),axis=0)
    # true=pred_exp

    # Pred_True_vision测试

    pred = np.load(r'F:\ENSO-Data\code-2\models\smivp\result_data\pred_list.npy')
    true = np.load(r'F:\ENSO-Data\code-2\models\smivp\result_data\output_list.npy')
    print(pred.shape)
    pred_true = Pred_True_vision(pred, true, path)
    acc, _ = pred_true.pers_nino_region('ido_west', False)
    # acc=[0.9736321964704332, 0.9434792855756601, 0.915116732790962, 0.881433036376515, 0.84929712832032, 0.8132301872133048, 0.7834861159094366, 0.7630605299326314, 0.7383100852935461, 0.7253610747682323, 0.6859132365718048, 0.6513792207310953, 0.6113214207360043, 0.5952699069588303, 0.57524611008731, 0.570729147589011, 0.5556539222439761, 0.5155572148806791, 0.5155662407745736, 0.5316992424433764, 0.5362978221620898, 0.5455812514487595, 0.5475807538939721, 0.5517100343473169]
    #
    #
    # # print(acc)
    # # def trainPlot(self,ACCList, iter, save_root,model_name,nino_region):
    pred_true.trainPlot(acc, 1, path, 'xx', 'ido_east')

    # test_vision_exp测试
    # vision=test_vision_exp(date_list,path)
    # vision.make_plt(pred,true)
    # # vision.visualize_u_v_sst(true[1],date_list[0])
    # vision.make_git()
    #
    # pre_val=np.load(r'/code/models/smivp/results/Debug/results/Debug/sv/preds.npy')
    # true_val=np.load(r'/code/models/smivp/results/Debug/results/Debug/sv/trues.npy')
    #
    # plt.imshow(pre_val[0,0,2,],cmap=c1)
    # plt.show()
    # print(pre_val.shape)
    # nino_region={0:[]}
    # val_cl=Pred_True_vision(pre_val,true_val,)

    # # 测试数据集中各区域
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    # pred=xr.open_zarr(r'F:\ENSO-Data\GODAS\godas_min.zarr',consolidated=True).sel(time=slice('2015-01-01','2016-12-01'))['godas_data']
    # lat=pred.coords['lat'].values
    # lon=pred.coords['lon'].values
    # print(lon.tolist())
    # print(lat.tolist())
    # nino34=[-4.5,5.5,190,240]
    # nino_4=[-4.5,5.5,210,270]
    # nino_3=[-4.5,5.5,160,210]
    # # IOD指数是通过计算印度洋西部（50°E～70°E, 10°S～10°N）与东部（90°E～110°E, 10°S～0°）区域平均海表温度距平之差来定义的。当IOD指数大于0
    # # .4
    # # 时
    # ido_weat=[-9.5,10.5,50,70]
    # ido_east=[-9.5,0.5,90,110]
    # nino_dict={0:nino34,1:nino_3,2:nino_4,3:ido_weat,4:ido_east}
    # # print(nino34[0:2])
    #
    #
    # for i in range(len(nino_dict)):
    #     lat_index = [np.where(lat == nino_dict[i][0])[0], np.where(lat == nino_dict[i][1])[0]]
    #     lon_index = [np.where(lon == nino_dict[i][2])[0], np.where(lon == nino_dict[i][3])[0]]
    #
    #     print(f'lat_nino34: {lat_index}; lon_nino34: {lon_index}')

    # nino34 = [-4.5, 5.5, 190, 240]
    # nino_4 = [-4.5, 5.5, 210, 270]
    # nino_3 = [-4.5, 5.5, 160, 210]
    # nino_dict = {0: nino34, 1: nino_3, 2: nino_4}

    # # 创建一个图表和轴对象
    # fig, ax = plt.subplots()
    #
    # # 设置矩形的左上角和右下角坐标
    # left_top = (lon_nino34[0][0],lat_nino34[1][0])  # 左上角坐标，范围是0到1，代表相对位置
    # right_bottom = (12, 14)  # 右下角坐标，范围是0到1
    #
    # # 创建一个矩形对象
    # rect = patches.Rectangle(left_top,10 ,2 , linewidth=1,
    #                          edgecolor='b', facecolor='none')
    #
    # # 将矩形添加到图表中
    # ax.add_patch(rect)
    #
    # # 设置图表的显示范围
    # ax.imshow(pre_val[0,0,2,],cmap=c1)
    #
    #
    # # 显示图表
    # plt.show()

    # import matplotlib.pyplot as plt
    #
    # # 示例数据
    # x = [1, 2, 3, 4, 5]
    # y = [2, 3, 5, 7, 11]
    #
    # # 绘制折线图
    # plt.plot(x, y, marker='o')  # marker='o'表示用圆圈标记数据点
    #
    # # # 在每个点旁边显示数值
    # for i in range(len(x)):
    #     plt.text(x[i], y[i], str(y[i]), ha='center', va='bottom')  # ha和va控制文本的水平和垂直对齐方式
    #
    # # 设置图表标题和坐标轴标签
    # plt.title('折线图示例')
    # plt.xlabel('X轴')
    # plt.ylabel('Y轴')
    #
    # # 显示图表
    # plt.show()
