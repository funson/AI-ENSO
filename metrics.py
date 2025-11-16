import numpy as np
import pytorch_ssim

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




