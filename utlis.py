import os
import logging
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path,preds_lst,trues_lst):
        score =val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print_log(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print_log(f'Validation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f}).  Saving model ...')
        #     plt_save = path + '/plt_dict'
        #     check_dir(plt_save)
        #     np.save(os.path.join(path, 'pred.npy'), preds_lst)
        #     np.save(os.path.join(path, 'true.npy'), trues_lst)
        #     np.save(os.path.join(path, 'input.npy'), input_list)
        torch.save(model.state_dict(), path+'/'+'checkpoints.pth')
        self.val_loss_min = val_loss

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
def score(y_pred, y_true, acc_weight):
    # for pytorch
    # acc_weight = np.array([1.5]*4 + [2]*7 + [3]*7 + [4]*6) * np.log(np.arange(24)+1)
    # acc_weight = torch.from_numpy(acc_weight).to(device)
    pred = y_pred - y_pred.mean(dim=0, keepdim=True)  # (N, 24)
    true = y_true - y_true.mean(dim=0, keepdim=True)  # (N, 24)
    cor = (pred * true).sum(dim=0) / (torch.sqrt(torch.sum(pred**2, dim=0) * torch.sum(true**2, dim=0)) + 1e-6)
    acc = (acc_weight * cor).sum()
    rmse = torch.mean((y_pred - y_true)**2, dim=0).sqrt().sum()
    return 2/3. * acc - rmse
if __name__ == '__main__':
    # 导入需要的库
    import time

    # 设置循环次数
    num_iterations = 10

    # 打开一个文件用于写入，如果文件不存在，将会被创建
    with open(r'F:\ENSO-Data\code\models\smivp\save_run\results-1-only-cmip6-oras\Debug\log.txt', 'w') as file:
        # 循环记录数值
        for i in range(num_iterations):
            # 假设记录的是当前时间戳
            timestamp = time.time()
            # 将时间戳转换为字符串，并写入文件，每条记录占一行
            file.write(f"{timestamp}\n")
            # 暂停一段时间，以便观察记录过程
            time.sleep(1)

    print("日志记录完成，内容已保存到log.txt文件中。")
