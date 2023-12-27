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
        data_loss = torch.mean(loss_h)
    return data_loss

# # Define Eval Function

def eval_2d_landslide_pad(model,
                          dataloader,
                          config,
                          device,
                          padding=0,
                          use_tqdm=True):
    model.eval()
    h_weight = config['train']['h_loss']
    u_weight = config['train']['u_loss']
    v_weight = config['train']['v_loss']
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    test_err = []

    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            batch_size, SX, SY, T, nfields = y.shape
            x_in = F.pad(x, (0, 0, 0, padding), "constant", 0)
            out = model(x_in).reshape(batch_size, SX, SY, T + padding, nfields)
            out = out[..., :-padding, :]
            data_loss = swe_loss(out, y, h_weight, u_weight, use_sum=False)

            test_err.append(data_loss.item())

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n')
    

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
# f5path='/data/home/scv6559/run/cyl/2d_t10.h5'
#Your app landslide case data
f5path_test='/data/home/scv6559/run/cyl/test2d_t10.h5'

testdata='testxy.txt'

print("read data")

# # Define the DataLoaders
print("Define the DataLoaders")
dataset3=DataLoader(testdata,f5path_test,gridx,gridy,gridt)
test_loader = torch.utils.data.DataLoader(dataset3, batch_size=config['test']['batchsize'], shuffle=False,num_workers=4)
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


optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.999), lr=config['train']['base_lr'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=config['train']['milestones'],
                                                 gamma=config['train']['scheduler_gamma'])

#Load from checkpoint
print("Load from checkpoint")
load_checkpoint(model, ckpt_path=config['test']['ckpt'], optimizer=None)
# # Evaluate Model
print("Evaluate Model")
eval_2d_landslide_pad(model,
                      test_loader,
                      config,
                      device,
                      padding=5,
                      use_tqdm=True)

# ## Padding
use_train_data = False
padding = 5
batch_size = config['test']['batchsize']
Nx = config['data']['nx']
Ny = config['data']['ny']
Nt = config['data']['nt'] + 1
Ntest = config['data']['n_test']
Ntrain = config['data']['n_train']
loader = test_loader
in_dim = config['model']['in_dim']
out_dim = config['model']['out_dim']

model.eval()
# model.to('cpu')
test_x = np.zeros((Ntest, Nx, Ny, Nt, in_dim))
preds_y = np.zeros((Ntest, Nx, Ny, Nt, out_dim))
test_y0 = np.zeros((Ntest, Nx, Ny, out_dim))
test_y = np.zeros((Ntest, Nx, Ny, Nt, out_dim))

with torch.no_grad():
    for i, data in enumerate(loader):
        data_x, data_y = data
        data_x, data_y = data_x.to(device), data_y.to(device)
        data_x_pad = F.pad(data_x, (0, 0, 0, padding), "constant", 0)
        pred_y_pad = model(data_x_pad).reshape(batch_size, Nx, Ny, Nt + padding, out_dim)
        pred_y = pred_y_pad[..., :-padding, :].reshape(data_y.shape)
        if i <= 100:
            test_x[i] = data_x.cpu().numpy()
            test_y[i] = data_y.cpu().numpy()
            test_y0[i] = data_x[..., 0, -out_dim:].cpu().numpy()  # same way as in training code
            preds_y[i] = pred_y.cpu().numpy()

x_size, y_size, t_size = preds_y.shape[1], preds_y.shape[2], preds_y.shape[3]
# y<0
for i in range(len(preds_y)):
    for time in range(t_size):
        for m in range(x_size):
            for n in range(y_size):
                if preds_y[i, m, n, time, 0] < 0:
                    preds_y[i, m, n, time, 0] = 0

# # Plot Results
key = -1
# key_t = (Nt - 3)
key_t = (Nt - 1) // 1


# X = X.reshape(Nx, Nx)

# # Save and Load Data
def save_data(data_path, test_x, test_y, preds_y, key_t):
    data_dir, data_filename = os.path.split(data_path)
    os.makedirs(data_dir, exist_ok=True)
    np.savez(data_path, test_x=test_x, test_y=test_y, preds_y=preds_y, key_t=key_t)


def load_data(data_path):
    data = np.load(data_path)
    test_x = data['test_x']
    test_y = data['test_y']
    preds_y = data['preds_y']
    key_t = int(data['key_t'])
    return test_x, test_y, preds_y, key_t


data_dir = 'data/2D_landslide'
data_filename = 'data.npz'
data_path = os.path.join(data_dir, data_filename)
# os.makedirs(data_dir, exist_ok=True)

save_data(data_path, test_x[:100], test_y[:100], preds_y[:100], key_t)

test_x, test_y, preds_y, key_t = load_data(data_path)

# output plt
for count in range(len(preds_y[:10])):
    with open(savefig_path + 'example_2d_{0:d}.dat'.format(count), 'w') as f:
        for t in range(preds_y.shape[3]):
            f.writelines('VARIABLES ="X","Y","Base","H","U", ''\n')
            output_data = preds_y[count].swapaxes(0, 1)
            # output_z = test_x[count].swapaxes(0,1)
            output_z = dataset3.search_z(count).cpu().numpy()
            rows, clos = output_data.shape[0], output_data.shape[1]
            f.writelines('Zone T="{}" i={},j={}'.format(t, rows, clos) + '\n')
            for clo in range(clos):
                for row in range(rows):
                    out_h = str(output_data[rows - 1 - row, clo, t, 0])
                    out_u = str(output_data[rows - 1 - row, clo, t, 1])
                    # out_v = str(output_data[rows-1-row, clo,t,2])
                    out_z = str(output_z[rows - 1 - row, clo, t])
                    f.writelines(str(xmin + dx * clo) + ' ' + str(
                        ymin + dy * row) + ' ' + out_z + ' ' + out_h + ' ' + out_u + ' ' + '\n')
    f.close()
# output h&u
out_txt = preds_y[0].swapaxes(0, 1)
rows, clos = out_txt.shape[0], out_txt.shape[1]

for t in range(out_txt.shape[2]):

    h_pred = out_txt[:, :, t, 0]
    with open(savefig_path + str(t) + str('h.txt'), "w") as f:
        f.writelines("ncols" + "   " + str(clos) + '\n')
        f.writelines("nrows" + "   " + str(rows) + '\n')
        f.writelines("xllcorner" + "   " + "0" + '\n')
        f.writelines("yllcorner" + "   " + "0" + '\n')
        f.writelines("cellsize" + "   " + str(dx) + '\n')
        f.writelines("NODATA_value" + "   " + "-9999.00000" + '\n')
        for row in range(rows):
            for clo in range(clos):
                f.writelines(str(h_pred[row][clo]) + ' ')
            f.write('\n')

    u_pred = out_txt[:, :, t, 1]
    with open(savefig_path + str(t) + str('u.txt'), "w") as f:
        f.writelines("ncols" + "   " + str(clos) + '\n')
        f.writelines("nrows" + "   " + str(rows) + '\n')
        f.writelines("xllcorner" + "   " + "0" + '\n')
        f.writelines("yllcorner" + "   " + "0" + '\n')
        f.writelines("cellsize" + "   " + str(dx) + '\n')
        f.writelines("NODATA_value" + "   " + "-9999.00000" + '\n')
        for row in range(rows):
            for clo in range(clos):
                f.writelines(str(u_pred[row][clo]) + ' ')
            f.write('\n')


def plot_predictions(key, key_t, test_x, test_y, preds_y, print_index=False, save_path=None, font_size=None, xmin=0,
                     xmax=1, ymin=0, ymax=1):
    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})

    Nsamples, Nx, Ny, Nt, Nfields = preds_y.shape
    h0 = torch.zeros((NX, Ny))
    pred_h = torch.zeros((NX, Ny))
    pred_u = torch.zeros((NX, Ny))
    # pred_v = torch.zeros((NX, Ny))
    true_h = torch.zeros((NX, Ny))
    true_u = torch.zeros((NX, Ny))
    # true_v = torch.zeros((NX, Ny))
    a = test_x[key]
    # Nt, Nx, _ = a.shape
    h0_i = a[..., 0, -1] * 141
    for i in range(Nx):
        for j in range(Ny):
            h0[i][Ny - 1 - j] = h0_i[i][j]
    # v0 = a[..., 0, -1]
    pred_h_i = preds_y[key, ..., key_t, 0]
    for i in range(Nx):
        for j in range(Ny):
            pred_h[i][Ny - 1 - j] = pred_h_i[i][j]
    pred_u_i = preds_y[key, ..., key_t, 1]
    for i in range(Nx):
        for j in range(Ny):
            pred_u[i][Ny - 1 - j] = pred_u_i[i][j]
    # pred_v_i = preds_y[key, ..., key_t, 2]
    # for i in range(Nx):
    #     for j in range(Ny):
    #       pred_v[i][Ny-1-j]=pred_v_i[i][j]
    true_h_i = test_y[key, ..., key_t, 0]
    for i in range(Nx):
        for j in range(Ny):
            true_h[i][Ny - 1 - j] = true_h_i[i][j]
    true_u_i = test_y[key, ..., key_t, 1]
    for i in range(Nx):
        for j in range(Ny):
            true_u[i][Ny - 1 - j] = true_u_i[i][j]
    # true_v_i = test_y[key, ..., key_t, 2]
    # for i in range(Nx):
    #     for j in range(Ny):
    #       true_v[i][Ny-1-j]=true_v_i[i][j]

    # T = a[:,:,2]
    # X = a[:,:,1]
    # x = X[0]
    x = torch.linspace(xmin, xmax, Nx + 1)[:-1]
    y = torch.linspace(ymin, ymax, Ny + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing='ij')
    u0 = torch.zeros_like(X)
    # v0 = torch.zeros_like(X)
    t = a[0, 0, key_t, 2] * 80

    grid_x, grid_y, grid_t = get_grid3d(Nx, Ny, Nt)

    fig = plt.figure(figsize=(24, 10))
    plt.subplot(2, 4, 1)

    plt.pcolormesh(X, Y, h0, cmap='jet', shading='gouraud')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Intial Condition $h(x,y)$')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.tight_layout()
    plt.axis("auto")

    ax1 = plt.subplot(2, 4, 2)
    # plt.pcolor(XX,TT, S_test, cmap='jet')
    p1 = plt.pcolormesh(X, Y, true_h, cmap='jet', shading='gouraud')
    c1 = p1.get_clim()
    p1.set_clim(c1)
    plt.colorbar(p1, ax=ax1)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    #     plt.title(f'Exact $\eta(x,y,t={t:.2f})$')
    plt.title(f'Exact $h(x,y,t={int(t)})$')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.tight_layout()
    plt.axis('auto')

    ax2 = plt.subplot(2, 4, 3)
    # plt.pcolor(XX,TT, S_pred, cmap='jet')
    plt.pcolormesh(X, Y, pred_h, cmap='jet', shading='gouraud')
    plt.colorbar(p1, ax=ax2)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    #     plt.title(f'Predict $\eta(x,y,t={t:.2f})$')
    plt.title(f'Predict $h(x,y,t={int(t)})$')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.axis('auto')

    plt.tight_layout()

    plt.subplot(2, 4, 4)
    # plt.pcolor(XX,TT, S_pred - S_test, cmap='jet')
    plt.pcolormesh(X, Y, pred_h - true_h, cmap='jet', shading='gouraud')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Absolute Error $h$')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.tight_layout()
    plt.axis('auto')

    plt.subplot(2, 4, 5)
    plt.pcolormesh(X, Y, u0, cmap='jet', shading='gouraud')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Intial Condition $u(x,y)$')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.tight_layout()
    plt.axis('auto')

    ax3 = plt.subplot(2, 4, 6)
    # plt.pcolor(XX,TT, S_test, cmap='jet')
    p2 = plt.pcolormesh(X, Y, true_u, cmap='jet', shading='gouraud')
    c2 = p2.get_clim()
    p2.set_clim(c2)
    plt.colorbar(p2, ax=ax3)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    #     plt.title(f'Exact $u(x,y,t={t:.2f})$')
    plt.title(f'Exact $u(x,y,t={int(t)})$')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.tight_layout()
    plt.axis('auto')

    ax4 = plt.subplot(2, 4, 7)
    # plt.pcolor(XX,TT, S_pred, cmap='jet')
    plt.pcolormesh(X, Y, pred_u, cmap='jet', shading='gouraud')
    plt.colorbar(p2, ax=ax4)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    #     plt.title(f'Predict $u(x,y,t={t:.2f})$')
    plt.title(f'Predict $u(x,y,t={int(t)})$')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.axis('auto')

    plt.tight_layout()

    ax5 = plt.subplot(2, 4, 8)
    # plt.pcolor(XX,TT, S_pred - S_test, cmap='jet')
    plt.pcolormesh(X, Y, pred_u - true_u, cmap='jet', shading='gouraud')
    plt.colorbar(p2, ax=ax5)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Absolute Error u')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.tight_layout()
    plt.axis('auto')
    if save_path is not None:
        plt.savefig(f'{save_path}.png', dpi=500, bbox_inches='tight')
    plt.show()


figures_dir = 'Landslide/figures/'
os.makedirs(figures_dir, exist_ok=True)
font_size = 16
for key in range(len(preds_y[:10])):
    save_path = os.path.join(figures_dir, f'SWE{key}')
    plot_predictions(key, key_t, test_x, test_y, preds_y, print_index=True, save_path=save_path, xmin=xmin, xmax=xmax,
                     ymin=ymin, ymax=ymax)


# ## Movies
def generate_movie_2D(key, test_x, test_y, preds_y, plot_title='', field=0, val_cbar_index=-1, err_cbar_index=-1,
                      val_clim=None, err_clim=None, font_size=None, movie_dir='',
                      movie_name='movie.gif', frame_basename='movie', frame_ext='jpg', remove_frames=True, xmin=0,
                      xmax=1, ymin=0, ymax=1):
    frame_files = []

    if movie_dir:
        os.makedirs(movie_dir, exist_ok=True)

    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})

    if len(preds_y.shape) == 4:
        Nsamples, Nx, Ny, Nt = preds_y.shape
        preds_y = preds_y.reshape(Nsamples, Nx, Ny, Nt, 1)
        test_y = test_y.reshape(Nsamples, Nx, Ny, Nt, 1)
    Nsamples, Nx, Ny, Nt, Nfields = preds_y.shape

    pred = torch.zeros((Nx, Ny, Nt))
    true = torch.zeros((Nx, Ny, Nt))

    pred_k = preds_y[key, ..., field]
    for t in range(Nt):
        for i in range(Nx):
            for j in range(Ny):
                pred[i][j][t] = pred_k[i][Ny - 1 - j][t]
    true_k = test_y[key, ..., field]
    for t in range(Nt):
        for i in range(Nx):
            for j in range(Ny):
                true[i][j][t] = true_k[i][Ny - 1 - j][t]
    error = pred - true

    a = test_x[key]
    x = torch.linspace(xmin, xmax, Nx + 1)[:-1]
    y = torch.linspace(ymin, ymax, Ny + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing='ij')
    t = a[0, 0, :, 2] * 80

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    pcm1 = ax1.pcolormesh(X, Y, true[..., val_cbar_index], cmap='jet', label='true', shading='gouraud')
    pcm2 = ax2.pcolormesh(X, Y, pred[..., val_cbar_index], cmap='jet', label='pred', shading='gouraud')
    pcm3 = ax3.pcolormesh(X, Y, error[..., err_cbar_index], cmap='jet', label='error', shading='gouraud')

    if val_clim is None:
        val_clim = pcm1.get_clim()
    if err_clim is None:
        err_clim = pcm3.get_clim()

    pcm1.set_clim(val_clim)
    plt.colorbar(pcm1, ax=ax1)
    ax1.axis('auto')

    pcm2.set_clim(val_clim)
    plt.colorbar(pcm2, ax=ax2)
    ax2.axis('auto')

    pcm3.set_clim(err_clim)
    plt.colorbar(pcm3, ax=ax3)
    ax3.axis('auto')

    plt.tight_layout(pad=3)

    for i in range(Nt):
        # Exact
        ax1.clear()
        pcm1 = ax1.pcolormesh(X, Y, true[..., i], cmap='jet', label='true', shading='gouraud')
        pcm1.set_clim(val_clim)
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$y$')
        ax1.set_title(f'Exact {plot_title}: $t={t[i]:.2f}$')
        ax1.axis('auto')

        # Predictions
        ax2.clear()
        pcm2 = ax2.pcolormesh(X, Y, pred[..., i], cmap='jet', label='pred', shading='gouraud')
        pcm2.set_clim(val_clim)
        ax2.set_xlabel('$x$')
        ax2.set_ylabel('$y$')
        ax2.set_title(f'Predict {plot_title}: $t={t[i]:.2f}$')
        ax2.axis('auto')

        # Error
        ax3.clear()
        pcm3 = ax3.pcolormesh(X, Y, error[..., i], cmap='jet', label='error', shading='gouraud')
        pcm3.set_clim(err_clim)
        ax3.set_xlabel('$x$')
        ax3.set_ylabel('$y$')
        ax3.set_title(f'Error {plot_title}: $t={t[i]:.2f}$')
        ax3.axis('auto')

        #         plt.tight_layout()
        fig.canvas.draw()

        if movie_dir:
            frame_path = os.path.join(movie_dir, f'{frame_basename}-{i:03}.{frame_ext}')
            frame_files.append(frame_path)
            plt.savefig(frame_path, dpi=500)

    if movie_dir:
        movie_path = os.path.join(movie_dir, movie_name)
        with imageio.get_writer(movie_path, mode='I') as writer:
            for frame in frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)

    if movie_dir and remove_frames:
        for frame in frame_files:
            try:
                os.remove(frame)
            except:
                pass


# ### Movie Parameters

key = 0
movie_dir = 'Landslide/movie/'
os.makedirs(movie_dir, exist_ok=True)

# ### Movie $\eta$

movie_name = '2D_landslide_h.gif'
frame_basename = '2D_landslide_h_frame'
frame_ext = 'jpg'
plot_title = "$h$"
field = 0
val_cbar_index = -1
err_cbar_index = -1
font_size = 16
remove_frames = True

generate_movie_2D(key, test_x, test_y, preds_y,
                  plot_title=plot_title,
                  field=field,
                  val_cbar_index=val_cbar_index,
                  err_cbar_index=err_cbar_index,
                  movie_dir=movie_dir,
                  movie_name=movie_name,
                  frame_basename=frame_basename,
                  frame_ext=frame_ext,
                  remove_frames=remove_frames,
                  font_size=font_size,
                  xmin=xmin,
                  xmax=xmax, ymin=ymin, ymax=ymax)

# ### Movie $u$

# get_ipython().run_line_magic('matplotlib', 'inline')
movie_name = '2D_landslide_u.gif'
frame_basename = '2D_landslide_u_frame'
frame_ext = 'jpg'
plot_title = "$u$"
field = 1
val_cbar_index = -1
err_cbar_index = -1
font_size = 16
remove_frames = True

generate_movie_2D(key, test_x, test_y, preds_y,
                  plot_title=plot_title,
                  field=field,
                  val_cbar_index=val_cbar_index,
                  err_cbar_index=err_cbar_index,
                  movie_dir=movie_dir,
                  movie_name=movie_name,
                  frame_basename=frame_basename,
                  frame_ext=frame_ext,
                  remove_frames=remove_frames,
                  font_size=font_size, xmin=xmin,
                  xmax=xmax, ymin=ymin, ymax=ymax)


# ### Movie $v$

# get_ipython().run_line_magic('matplotlib', 'inline')
# movie_name = 'SWE_Nonlinear_v.gif'
# frame_basename = 'SWE_Nonlinear_v_frame'
# frame_ext = 'jpg'
# plot_title = "$v$"
# field = 2
# val_cbar_index = -1
# err_cbar_index = -1
# font_size = 12
# remove_frames = True
#
# generate_movie_2D(key, test_x, test_y, preds_y,
#                   plot_title=plot_title,
#                   field=field,
#                   val_cbar_index=val_cbar_index,
#                   err_cbar_index=err_cbar_index,
#                   movie_dir=movie_dir,
#                   movie_name=movie_name,
#                   frame_basename=frame_basename,
#                   frame_ext=frame_ext,
#                   remove_frames=remove_frames,
#                   font_size=font_size,xmin=xmin,
#                   xmax=xmax,ymin=ymin,ymax=ymax)

print("over")


