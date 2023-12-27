#!/usr/bin/env python
# coding: utf-8
from IPython.core.display import display, HTML
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.utils.data import DataLoader
import h5py
from models import FNN3d
from train_utils import Adam
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("log")
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from train_utils.utils import save_checkpoint, get_grid3d,load_checkpoint, load_config
from train_utils.losses import LpLoss
from torch.utils.data import Dataset
import imageio
def swe_loss(s_pred, s_true, h_weight, u_weight, use_sum=True):
    h_pred = s_pred[..., 0]
    u_pred = s_pred[..., 1]
    # v_pred = s_pred[..., 2]
    h_true = s_true[..., 0]
    u_true = s_true[..., 1]
    # v_true = s_true[..., 2]
    lploss = LpLoss(size_average=True)
    loss_h = lploss(h_pred, h_true)
    loss_u = lploss(u_pred, u_true)
    # loss_v = lploss(v_pred, v_true)
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
def train_2d_landslide_pad(model,
                           train_loader,
                           dataloader,
                           device,
                           optimizer, scheduler,
                           config,
                           padding=0,
                           use_tqdm=True):
    data_weight = config['train']['xy_loss']
    h_weight = config['train']['h_loss']
    u_weight = config['train']['u_loss']
    v_weight = config['train']['v_loss']
    ckpt_freq = config['train']['ckpt_freq']

    model.train()
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
            batch_size, SX, SY, T, nout = y.shape[0], y.shape[1], y.shape[2], y.shape[3], y.shape[4]
            out = model(x_in).reshape(batch_size, SX, SY, T + padding, nout)

            out = out[..., :-padding, :]
            data_loss = swe_loss(out, y, h_weight, u_weight)

            total_loss = (data_loss * data_weight)
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
                    f'data l2 error: {data_l2:.5f}; '
                )
            )
        writer.add_scalar('Loss/train',
                          train_loss,
                          e)

        if e % ckpt_freq == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
        if e % 5 == 0:
            model.eval()
            test_err = []

            with torch.no_grad():
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    batch_size, SX, SY, T, nout = y.shape
                    x_in = F.pad(x, (0, 0, 0, padding), "constant", 0)
                    out = model(x_in).reshape(batch_size, SX, SY, T + padding, nout)
                    out = out[..., :-padding, :]
                    data_loss = swe_loss(out, y, h_weight, u_weight, use_sum=False)
                    test_err.append(data_loss.item())

            mean_err = np.mean(test_err)
            std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

            print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n')

            writer.add_scalars('Loss', {'train loss': train_loss, 'valid loss': mean_err}, e)

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    writer.close()

class DataLoader(Dataset):
    def __init__(self, datapath, h5file,gridx,gridy,gridt):
        self.datapath = datapath
        self.h5file = h5file
        self.x_list = []
        self.gridx=gridx
        self.gridy=gridy
        self.gridt=gridt
        self.NX=self.gridx.shape[2]
        self.NY=self.gridx.shape[1]
        self.T=self.gridx.shape[3]

        with open(self.datapath, 'r') as f_txt:
            for line in f_txt.readlines():
                xpath  = line.split(' ')[0]
                self.x_list.append(xpath)

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        f_h5 = h5py.File(self.h5file, 'r')
        xname = self.x_list[idx]
        data = torch.tensor(np.array(f_h5[xname])).float()
        data=torch.cat([self.gridy, self.gridx,self.gridt,data], dim=-1)

        self.data=data.squeeze(0).permute(1,0,2,3)#(nx,ny,nt,nint[y,x,t,fai,z,h,u])
        self.data[:,:,:,4]=(self.data[:,:,:,4]-torch.min(self.data[:,:,0,4]))/2133#delta_z_max

        # The height of the source was normalized and the maximum height in the dataset was 141m
        h0=self.data[:,:, 0, 5].reshape(self.NX,self.NY)/141
        h0 = h0.unsqueeze(2).unsqueeze(3).repeat([1, 1, self.T, 1])#The initial height is proposed as input
        self.xdata=torch.cat([self.data[:,:,:,:5],h0],dim=-1)

        self.ydata = self.data[:,:,:,5:7]
        f_h5.close()
        return self.xdata, self.ydata
    def search_z(self,idx):
        f_h5 = h5py.File(self.h5file, 'r')
        xname = self.x_list[idx]
        data = torch.tensor(np.array(f_h5[xname])).float()
        self.data=data.squeeze(0)#(ny,nx,nt,nin[fai,z,h,u])
        z=self.data[:,:,:,1]
        f_h5.close()
        return z

# Load Configuration File
config_file = 'configs/custom/2D_landslide-0000.yaml'
config = load_config(config_file)
display(config)

# Define Parameters
Nsamples = config['data']['total_num']
NX = config['data']['nx']
NY = config['data']['ny']
Nt = config['data']['nt']
savefig_path=config['data']['savefig_path']
batchsize=config['train']['batchsize']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Lx = 2500 #Actual length
Ly=2500
dx=Lx/NX
dy=Ly/NY
tend = 1
xmin=0
xmax=2500
ymin=0
ymax=2500
T=Nt+1

#Both temporal and spatial coordinates are normalized
gridx = torch.tensor(np.linspace(xmin, 1, NX + 1)[:-1], dtype=torch.float)
gridx = gridx.reshape(1, 1, NX, 1, 1).repeat([1, NY, 1, T, 1])

gridy = torch.tensor(np.linspace(ymin, 1, NY + 1)[:-1], dtype=torch.float)
gridy = gridy.reshape(1, NY, 1, 1, 1).repeat([1, 1, NX, T, 1])

gridt = torch.tensor(np.linspace(0, tend, T), dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, NY, NX, 1, 1])

#Reading data files

#Training and testing data with a time interval of 10s
f5path='/data/home/scv6559/run/cyl/2d_t10.h5'

traindata='trainxy.txt'
validdata='validxy.txt'

print("read data")

# # Define the DataLoaders
print("Define the DataLoaders")
dataset1=DataLoader(traindata,f5path,gridx,gridy,gridt)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=config['train']['batchsize'], shuffle=False,num_workers=4)
dataset2=DataLoader(validdata,f5path,gridx,gridy,gridt)
valid_loader = torch.utils.data.DataLoader(dataset2, batch_size=config['valid']['batchsize'], shuffle=False,num_workers=4)
# # Define the Model
model = FNN3d(modes1=config['model']['modes1'],
              modes2=config['model']['modes2'],
              modes3=config['model']['modes3'],
              fc_dim=config['model']['fc_dim'],
              layers=config['model']['layers'],
              in_dim=config['model']['in_dim'],
              out_dim=config['model']['out_dim'],
              activation=config['model']['activation'],
             ).to(device)
print("Load from checkpoint")
# load_checkpoint(model, ckpt_path=config['train']['ckpt'])

#fine tune
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.fc1.parameters():
# 	param.requires_grad = True
# for param in model.fc2.parameters():
# 	param.requires_grad = True
# for param in model.fc0.parameters():
# 	param.requires_grad = True
# for param in model.sp_convs[2].parameters():
# 	param.requires_grad = True
# # for param in model.ws[2].parameters():
# # 	param.requires_grad = True
# for param in model.sp_convs[3].parameters():
# 	param.requires_grad = True
# # for param in model.ws[3].parameters():
# # 	param.requires_grad = True
# for k, v in model.named_parameters():
#     if v.requires_grad:
#         print("可训练的模型参数名称 {}".format(k))
#     else:
#         print("已被冻结的模型参数名称 {}".format(k))


optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.999), lr=config['train']['base_lr'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=config['train']['milestones'],
                                                 gamma=config['train']['scheduler_gamma'])

#Load from checkpoint

#load_checkpoint(model, ckpt_path=config['train']['ckpt'], optimizer=None)

# # Train the Model
print("Train the Model")
train_2d_landslide_pad(model,
                  train_loader,
                  valid_loader,
                  device,
                  optimizer,
                  scheduler,
                  config,
                  padding=5,
                  tend=tend)