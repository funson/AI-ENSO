import numpy as np
import pytorch_ssim
from skimage.metrics import structural_similarity as cal_ssim

def MAE(pred, true):
    return np.mean(np.abs(pred-true),axis=(0,1)).sum()

def MSE(pred, true):
    return np.mean((pred-true)**2,axis=(0,1)).sum()

# cite the `PSNR` code from E3d-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py line 39-40
def PSNR(mse,max):
    return 20 * np.log10(max) - 10 * np.log10(mse)

def RMSE(pred, true):
    rmse = np.mean((pred - true) ** 2, axis=(2, 3))
    rmse = np.sum(np.sqrt(rmse).mean(axis=0))
    return rmse


def metric(pred, true,  return_ssim_psnr=False, clip_range=[0, 1]):

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse=RMSE(pred, true)


    return mse, mae,rmse

def metric_TRUE(pred, true, mean, std, return_ssim_psnr=False, clip_range=[0, 1]):
    pred = pred*std + mean
    true = true*std + mean
    mae = MAE(pred, true)
    mse = MSE(pred, true)

    if return_ssim_psnr:
        pred = np.maximum(pred, clip_range[0])
        pred = np.minimum(pred, clip_range[1])
        ssim, psnr = 0, 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2), true[b, f].swapaxes(0, 2), multichannel=True)
                psnr += PSNR(pred[b, f], true[b, f])
        ssim = ssim / (pred.shape[0] * pred.shape[1])
        psnr = psnr / (pred.shape[0] * pred.shape[1])
        return mse, mae, ssim, psnr
    else:
        return mse, mae


