import os
import torch
from models import FNN2d, FNN2d_AD
from train_utils import Adam
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("log")
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from train_utils.utils import save_checkpoint, load_checkpoint, load_config, update_config
from train_utils.losses import LpLoss
import imageio
from IPython.core.display import display, HTML
import h5py
from torch.utils.data import Dataset

def swe_loss(s_pred, s_true, h_weight, u_weight, use_sum=True):
    h_pred = s_pred[..., 0]
    u_pred = s_pred[..., 1]
    h_true = s_true[..., 0]
    u_true = s_true[..., 1]
    lploss = LpLoss(size_average=True)
    loss_h = lploss(h_pred, h_true)
    loss_u = lploss(u_pred, u_true)
    if use_sum:
        loss_h *= h_weight
        loss_u *= u_weight
    loss_s = torch.stack([loss_h, loss_u], dim=-1)
    if use_sum:
        data_loss = torch.sum(loss_s)
    else:
        data_loss = torch.mean(loss_s)
    return data_loss
# # Define Training Fuction

def train_1d_padding(model,
                          device,
                          train_loader,
                          dataloader,
                          optimizer,
                          scheduler,
                          config,
                          padding=0,
                          use_tqdm=True):


    data_weight = config['train']['xy_loss']
    h_weight = config['train']['h_loss']
    u_weight = config['train']['u_loss']
    ckpt_freq = config['train']['ckpt_freq']
    model.train()
    myloss = LpLoss(size_average=True)

    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    for e in pbar:
        model.train()
        data_l2 = 0.0
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_in = F.pad(x, (0, 0, 0, padding), "constant", 0)
            batch_size, S, T,nout = y.shape[0],y.shape[1],y.shape[2],y.shape[3]
            new_shape = (batch_size, S,T + padding)
            out = model(x_in).reshape(batch_size, S,T + padding,nout)
            out = out[..., :-padding, :]

            data_loss = swe_loss(out, y,h_weight,u_weight)
            total_loss =  data_loss * data_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_loss += total_loss.item()
        scheduler.step()
        data_l2 /= len(train_loader)
        train_loss /= len(train_loader)
        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {e}, train loss: {train_loss:.5f} '
                    f'data l2 error: {data_l2:.5f}'
                )
            )
        if e % ckpt_freq == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
        if e % 5 == 0:
            model.eval()
            myloss = LpLoss(size_average=True)

            test_err = []
            f_err = []
            with torch.no_grad():
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    x_in = F.pad(x, (0, 0, 0, padding), "constant", 0)
                    batch_size, S, T, nout = old_shape = y.shape[0], y.shape[1], y.shape[2], y.shape[3]
                    new_shape = (batch_size, S, T + padding)
                    out = model(x_in).reshape(batch_size, S, T + padding, nout)
                    out = out[..., :-padding, :]
                    data_loss = swe_loss(out, y,h_weight,u_weight,use_sum=False)

                    test_err.append(data_loss.item())

            mean_err = np.mean(test_err)
            std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

            print(f'Epoch {e},==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n')
            writer.add_scalars('Loss',{'train loss':train_loss,'valid loss':mean_err},e)

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    writer.close()



# # Load Configuration File
config_file = 'configs/custom/1d-0000.yaml'
config = load_config(config_file)
display(config)
# # Define Parameters
Nsamples = config['data']['total_num']
N = config['data']['nx']
Nt0 = config['data']['nt']
sub_x = config['data']['sub']
sub_t = config['data']['sub_t']
Nx = N // sub_x
Nt = Nt0 // sub_t + 1
dim = 1
L = 2500
dx=L/N
tend = 80
n=0.0
xmin=0
xmax=2500
savefig_path=config['data']['savefig_path']
batchsize=config['train']['batchsize']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
f5path='/data/home/scv6559/run/cyl/1d_t10.h5'

traindata='trainxy.txt'
validdata='validxy.txt'

gridx = torch.tensor(np.linspace(xmin, 1, Nx + 1)[:-1], dtype=torch.float)
gridx = gridx.reshape(1, Nx, 1, 1).repeat([1, 1, Nt, 1])

gridt = torch.tensor(np.linspace(0, 1, Nt), dtype=torch.float)
gridt = gridt.reshape(1, 1, Nt, 1).repeat([1, Nx, 1, 1])
print("read data")

# Define the DataLoaders
print("Define the DataLoaders")

dataset1=DataLoader(traindata,f5path,gridx,gridt)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=config['train']['batchsize'], shuffle=True)
dataset2=DataLoader(validdata,f5path,gridx,gridt)
valid_loader = torch.utils.data.DataLoader(dataset2, batch_size=config['valid']['batchsize'], shuffle=False)


model = FNN2d(modes1=config['model']['modes1'],
                   modes2=config['model']['modes2'],
                   fc_dim=config['model']['fc_dim'],
                   layers=config['model']['layers'],
                   activation=config['model']['activation']).to(device)

optimizer = Adam(model.parameters(), betas=(0.9, 0.999),lr=config['train']['base_lr'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=config['train']['milestones'],
                                                 gamma=config['train']['scheduler_gamma'])

#Load from checkpoint
print("Load from checkpoint")
#load_checkpoint(model, ckpt_path=config['test']['ckpt'], optimizer=None)
# # Train the Model
print("Train the Model")
train_1d_padding(model, device,
              train_loader,
              valid_loader,
              optimizer,
              scheduler,
              config,
              padding=5)
