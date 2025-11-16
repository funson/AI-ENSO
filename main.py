import argparse
from exp import Exp
import torch
import warnings
warnings.filterwarnings('ignore')

#SODA version 2.2.4: https://climatedataguide.ucar.edu/climate-data/soda-simple-ocean-dataassimilation
# GODAS: https://www.esrl.noaa.gov/psd/data/gridded/data.godas.html
# The CMIP6 database: https://esgf-node.llnl.gov/projects/cmip6/.
# It is necessary to download the corresponding data from the above links.
# The data processing steps are as follows: first, unify the spatial size of the specified region, then remove the long-term trend and perform standardization.
# These tasks were not implemented using Python programming scripts but with CDO scripts, so the relevant scripts cannot be provided.
train_loc = r'./data_enso/train_16_model_data_cl_t_s_stand_v1.nc' #(diff_models,t,v,h,w)
vail_soda_loc = r'./data_enso/soda_val_stand_v1.nc'#(t,v,h,w)
test_godas_loc = r'./data_enso/godas_test_stand_v2.nc' #(t,v,h,w)


path = {0: fr"{train_loc}", 1: fr"{vail_soda_loc}", 2: fr"{test_godas_loc}"}


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, )
    parser.add_argument('--res_dir', default='./results-godas-cmip6-mse-7_no_godas_train', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_train_type', default='cmip6', type=str,choices=['cmip6','soda','godas'])
    parser.add_argument('--model_name', default='Swin-Transformer', type=str,choices=['cmip6','soda','godas'])

    # dataset parameters
    parser.add_argument('--train_batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--test_batch_size', default=3, type=int, help='Batch size')
    parser.add_argument('--size', default=[12,20], type=list, help='input-output_len')
    parser.add_argument('--val_batch_size', default=5, type=int, help='Batch size')
    parser.add_argument('--data_path', default=path)
    parser.add_argument('--dataname', default='cmip6', choices=['soda', 'godas'])
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--Pre_Train', default=False, type=bool)
    parser.add_argument('--pred_mode_path', default='./results/Debug', type=str, help='path to pred model')
    parser.add_argument('--shuffle', default=False, type=bool)
    parser.add_argument('--val_shuffle', default=True, type=bool)

    parser.add_argument('--total_len', default=12, type=int)
    parser.add_argument('--teacher_forcing', default=False, type=bool)
    parser.add_argument('--his_frames', default=12, type=int)
    parser.add_argument('--pre_len', default=20, type=int)
    parser.add_argument('--input_frames', default=3, type=int)
    parser.add_argument('--img_size', default=[60, 240], type=list)
    parser.add_argument('--patch_size', default=2,type=int)
    parser.add_argument('--in_chans', default=4,type=int)
    parser.add_argument('--embed_dim', default=96,type=int)
    parser.add_argument('--depths', default=[2,6],type=list)
    parser.add_argument('--num_heads', default=[8,16],type=list)
    parser.add_argument('--up_depths', default=2,type=int)
    parser.add_argument('--window_size', default=[3,5],type=list)
    parser.add_argument('--mlp_ratio', default=0.1,type=int)
    parser.add_argument('--qkv_bias', default=True,type=bool)
    parser.add_argument('--drop_rate', default=0.1,type=float)
    parser.add_argument('--attn_drop_rate', default=0.01,type=float)
    parser.add_argument('--drop_path_rate', default=0.01,type=float)
    parser.add_argument('--ape', default=True,type=bool)
    parser.add_argument('--patch_norm', default=True,type=bool)
    parser.add_argument('--use_checkpoint', default=False,type=bool)
    parser.add_argument('--pretrained_window_sizes', default=[0, 0, 0, 0],type=list)

    # Training parameters
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--patience', default=3, type=int, help='patience for steps')
    parser.add_argument('--loss-type', default='', type=str, help='loss function')
    parser.add_argument('--mse_weight', default=0.8, type=int, help='loss mse_weight')
    parser.add_argument('--rmse_sst_weight', default=0.2, type=int, help='loss rmse_sst_weight')

    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    config['loss-type']=f'mse*{config["mse_weight"]}+rmse_sst*{config["rmse_sst_weight"]}'

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)
