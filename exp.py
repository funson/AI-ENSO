import os.path as osp
import pickle

from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from data_set.dataset import *
from tools.metrics import *
from tools.recorder import *
from Model.LSTA_Swin import LSTA_Swin
from utlis import *
from tensorboardX import SummaryWriter

writer = SummaryWriter()
class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()
        self.writer = SummaryWriter(f'./{self.path}')

        self._preparation()

        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()
        self.weight = torch.from_numpy(np.array([1.5]*4 + [2]*7 + [3]*7 + [4]*6) * np.log(np.arange(24)+1)).to(self.args.device)
    def score(self, y_pred, y_true):
        with torch.no_grad():
            sc = score(y_pred, y_true, self.weight)
        return sc.item()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        # check_dir(self.args.pred_mode_path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the models
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = LSTA_Swin(
            total_len=self.args.total_len,
            teacher_forcing=True,
            pre_len=self.args.pre_len,
            his_frames=self.args.his_frames,
            input_frames=self.args.input_frames,
            img_size=self.args.img_size,
            patch_size=self.args.patch_size,
            in_chans=self.args.in_chans,
            embed_dim=self.args.embed_dim,
            depths=self.args.depths,
            num_heads=self.args.num_heads,
            up_depths=self.args.up_depths,
            window_size=self.args.window_size,
            mlp_ratio=self.args.mlp_ratio,
            qkv_bias=self.args.qkv_bias,
            drop_rate=self.args.drop_rate,
            attn_drop_rate=self.args.attn_drop_rate,
            drop_path_rate=self.args.drop_path_rate,
            ape=self.args.ape, patch_norm=self.args.patch_norm,
            use_checkpoint=self.args.use_checkpoint,
            pretrained_window_sizes=self.args.pretrained_window_sizes,
            N_T=self.args.N_T
            ).to(self.device)


    def _get_data(self):
        config = self.args.__dict__


        self.train_dataset= CMIP_Dataset(Type_map='train',data_path=self.args.data_path,size=self.args.size)

        train_dataset, test_dataset = random_split(self.train_dataset, [int(len(self.train_dataset)*0.8), len(self.train_dataset)-int(len(self.train_dataset)*0.8)])

        # soda
        # self.train_dataset= SODA_Dataset(Type_map='train',data_path=self.args.data_path,size=self.args.size)
        # self.val_dataset= SODA_Dataset(Type_map='val',data_path=self.args.data_path,size=self.args.size)

        # godas
        # self.train_dataset= GODAS_Dataset(Type_map='train',data_path=self.args.data_path,size=self.args.size)
        # self.val_dataset= GODAS_Dataset(Type_map='val',data_path=self.args.data_path,size=self.args.size)





        self.train_loader=DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=self.args.shuffle,
            num_workers=self.args.num_workers,
        drop_last=True
        )

        self.test_loader=DataLoader(
            test_dataset,
            batch_size=self.args.test_batch_size,
            shuffle=self.args.val_shuffle,
            num_workers=self.args.num_workers,
            drop_last=True
        )


    def _select_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.lr)

        if self.args.data_train_type=='cmip6':
            total_len=len(self.train_loader)

        else:
            total_len=len(self.train_loader)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.args.lr,
            steps_per_epoch=total_len,
            epochs=self.args.epochs,
        )
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def loss_sst(self, y_pred, y_true):
        loss=(y_pred - y_true)
        loss[loss<0.2]*=0.95
        loss[(loss>=0.2)&(loss<1)]*=1.25
        loss[(loss>=1)&(loss<2.5)]*=2.55
        loss[(loss>=2.5)&(loss<3.5)]*=3.75
        loss[loss>=3.5]*=4.55

        rmse = torch.mean((y_pred - y_true)**2, dim=[2, 3])
        rmse = torch.sum(rmse.sqrt().mean(dim=0))
        return rmse

    def mse_loss(self,input,target):
        #input (B,T,V,H,W)
        nino34 = [25, 35, 70, 180]
        # print(target.shape,input.shape)
        loss=self.criterion(target,input)

        loss_nino34 = loss[:,:,nino34[0]:nino34[1]+1,nino34[2]:nino34[3]+1]
        loss_nino34[loss_nino34<0.2]*=0.95
        loss_nino34[(loss_nino34>=0.2)&(loss_nino34<1.0)]*=1.55
        loss_nino34[(loss_nino34>=1.0)&(loss_nino34<2.0)]*=2.55
        loss_nino34[(loss_nino34>=2.0)&(loss_nino34<3.5)]*=3.25
        # torch.nan
        loss_nino34[loss_nino34>=3.5]*=4.55

        loss[:,:,nino34[0]:nino34[1]+1,nino34[2]:nino34[3]+1]=loss_nino34

        mse=torch.mean(loss)
        return mse



    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        ema = EMA(self.model, 0.999)
        ema.register()

        if self.args.Pre_Train==True:
            model_static=torch.load(self.args.pred_mode_path+'//'+'checkpoints.pth')
            self.model.load_state_dict(model_static)




        val_nino34_avg=0
        for epoch in range(config['epochs']):
            # print('Epoch {}/{}'.format(epoch+1, config['epochs']))
            self.model.train().float()
            train_loader = [self.train_loader]


            train_loss_list=[]
            train_rmse_loss_list=[]
            for loader in range(len(train_loader)):

                train_pbar = tqdm(train_loader[loader])

                for batch_x, batch_y in train_pbar:
                    self.optimizer.zero_grad()

                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred_y = self.model(batch_x)

                    mse_loss = self.mse_loss(pred_y, batch_y)
                    rmse_loss=self.loss_sst(pred_y, batch_y)
                    loss=mse_loss*self.args.mse_weight+rmse_loss*self.args.rmse_sst_weight
                    # loss=torch.mean(self.criterion(pred_y,batch_y))

                    train_loss_list.append(mse_loss.item())
                    train_rmse_loss_list.append(rmse_loss.item())
                    train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    ema.update()

            train_loss = np.average(train_loss_list)
            train_rmse_loss = np.average(train_rmse_loss_list)
            ema.apply_shadow()

            self.writer.add_scalar('train_loss', train_loss, epoch)
            self.writer.add_scalar('train_rmse_loss', train_rmse_loss, epoch)

            if epoch % args.log_step == 0:
                with torch.no_grad():

                    vali_loss, score, nino34_loss, vali_rmse_loss, val_mse_loss, nino34_avg, nino3_avg, nino4_avg, ido_avg ,preds_lst,trues_lst= self.vali(
                        self.test_loader, epoch, ema.model)
                    ema.restore()

                    print_log(f'nino34_ema: {nino34_avg}')
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))

                    print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f} nino34_loss:{3:.4f} val_rmse_loss:{4:.4f}"
                              " val_mse_loss:{5:.4f} nino34_avg:{6:.4f} nino3_avg:{7:.4f} nino4_avg:{8:.4f} ido_avg:{9:.4f} \n".format(
                            epoch + 1, train_loss, vali_loss, nino34_loss, vali_rmse_loss, val_mse_loss, nino34_avg,
                            nino3_avg, nino4_avg, ido_avg))
            #
                early_stopping(nino34_avg, ema.model, self.path,preds_lst,trues_lst)
                if     val_nino34_avg<nino34_avg:
                    val_nino34_avg =nino34_avg
                    print_log('>>>>>>save<<<<<<<')


                    best_model_path = self.path + '/' + f'checkpoints.pth'
                    self.model.load_state_dict(torch.load(best_model_path))
                else:
                    print_log('>>>>>>no save<<<<<<<')

                if early_stopping.early_stop:
                    break



        return self.model

    def vali(self, vali_loader, iter, model):
        model.eval()
        batch_x_list, preds_lst, trues_lst, total_loss, total_rmse_loss, total_mse_loss = [], [], [], [], [], []
        vali_pbar = tqdm(vali_loader)

        for i, (batch_x, batch_y) in enumerate(vali_pbar):


            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = model(batch_x)


            preds_lst.append(pred_y[:,:,2].detach().cpu().numpy())
            trues_lst.append(batch_y[:,:,2].detach().cpu().numpy())

            mse_loss = self.mse_loss(pred_y, batch_y)
            rmse_loss = self.loss_sst(pred_y, batch_y)
            loss = mse_loss * self.args.mse_weight + rmse_loss * self.args.rmse_sst_weight
            # loss =torch.mean( self.criterion(pred_y, batch_y))

            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())
            total_rmse_loss.append(rmse_loss.item())
            total_mse_loss.append(mse_loss.item())

        total_loss = np.average(total_loss)
        total_rmse_loss = np.average(total_rmse_loss)
        total_mse_loss = np.average(total_mse_loss)


        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)

        plt_save = self.path + '/plt_dict'
        check_dir(plt_save)


        vision = Pred_True_vision(preds, trues, self.path)
        score_34, nino34_loss, nino34_person, _ = vision.pers_nino_region('nino34', score_acu=True)
        score_4, nino4_loss, nino4_person, _ = vision.pers_nino_region('nino4', score_acu=True)
        score_3, nino3_loss, nino3_person, _ = vision.pers_nino_region('nino3', score_acu=True)
        score_ido, ido_loss, ido_person, _ = vision.pers_nino_region('ido', score_acu=True)
        nino34_avg = sum(nino34_person) / len(nino34_person)
        nino3_avg = sum(nino3_person) / len(nino4_person)
        nino4_avg = sum(nino4_person) / len(nino34_person)
        ido_avg = sum(ido_person) / len(nino34_person)

        vision.trainPlot(nino34_person, iter, plt_save, self.args.model_name, 'nino34')
        vision.trainPlot(nino4_person, iter, plt_save, self.args.model_name, 'nino4')
        vision.trainPlot(nino3_person, iter, plt_save, self.args.model_name, 'nino3')
        vision.trainPlot(ido_person, iter, plt_save, self.args.model_name, 'ido')

        model.train()
        return total_loss, score_34, nino34_loss, total_rmse_loss, total_mse_loss, nino34_avg, nino3_avg, nino4_avg, ido_avg,preds,trues

    def test(self, args):
        model_static=torch.load(os.path.join(self.args.pred_mode_path,'checkpoints.pth'))
        self.model.load_state_dict(model_static)
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in self.test_loader:
            pred_y = self.model(batch_x.to(self.device))

            inputs_lst.append(batch_x.detach().cpu().numpy())
            trues_lst.append(batch_y.detach().cpu().numpy())
            preds_lst.append(pred_y.detach().cpu().numpy())


        preds= np.concatenate(preds_lst,axis=0)
        inputs= np.concatenate(inputs_lst,axis=0)
        trues= np.concatenate(trues_lst,axis=0)
        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mse, mae = metric(preds, trues,  False)
        # print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return mse
