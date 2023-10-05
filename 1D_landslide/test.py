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
        data_loss = torch.mean(loss_h)
    return data_loss

# # Define Eval Function
def eval_1d_padding(model,
                         dataloader,
                         config,
                         padding=0,
                         device=None,
                         use_tqdm=True):
    model.eval()
    h_weight = config['train']['h_loss']
    u_weight = config['train']['u_loss']
    myloss = LpLoss(size_average=True)

    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    test_err = []
    f_err = []
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            x_in = F.pad(x, (0, 0, 0, padding), "constant", 0)
            batch_size, S,T ,nout= old_shape = y.shape[0],y.shape[1],y.shape[2],y.shape[3]
            new_shape = (batch_size, S,T + padding)
            out = model(x_in).reshape(batch_size, S,T + padding,nout )
            out = out[..., :-padding, :]
            #             out = model(x).reshape(y.shape)
            data_loss = swe_loss(out, y,h_weight, u_weight,use_sum=False)

            test_err.append(data_loss.item())

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n')

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
testpath='/data/home/scv6559/run/cyl/test1d_t10.h5'

testdata='testxy.txt'

gridx = torch.tensor(np.linspace(xmin, 1, Nx + 1)[:-1], dtype=torch.float)
gridx = gridx.reshape(1, Nx, 1, 1).repeat([1, 1, Nt, 1])

gridt = torch.tensor(np.linspace(0, 1, Nt), dtype=torch.float)
gridt = gridt.reshape(1, 1, Nt, 1).repeat([1, Nx, 1, 1])
print("read data")

# Define the DataLoaders
print("Define the DataLoaders")

dataset3=DataLoader(testdata,testpath,gridx,gridt)
test_loader = torch.utils.data.DataLoader(dataset3, batch_size=config['test']['batchsize'], shuffle=False)


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
load_checkpoint(model, ckpt_path=config['test']['ckpt'], optimizer=None)

# # Evaluate Model
print("Evaluate Model")
eval_1d_padding(model, test_loader, config=config, padding=5,device=device)

# # Generate Test Predictions
use_train_data = False
padding = 5
batch_size = config['test']['batchsize']
Nx = config['data']['nx']
# Ny = config['data']['nx']
Nt = config['data']['nt'] + 1
Ntest = config['data']['n_test']
Ntrain = config['data']['n_train']
loader = test_loader
if use_train_data:
    Ntest = Ntrain
    loader = train_loader
in_dim = config['model']['in_dim']
out_dim = config['model']['out_dim']

model.eval()
# model.to('cpu')
test_x = np.zeros((Ntest,Nx,Nt,in_dim))
preds_y = np.zeros((Ntest,Nx,Nt,out_dim))
test_y = np.zeros((Ntest,Nx,Nt,out_dim))


with torch.no_grad():
    for i, data in enumerate(loader):
        data_x, data_y = data
        data_x, data_y = data_x.to(device), data_y.to(device)
        data_x_pad = F.pad(data_x, (0, 0, 0, padding), "constant", 0)
        pred_y_pad = model(data_x_pad).reshape(batch_size,Nx, Nt + padding,out_dim)
        pred_y = pred_y_pad[..., :-padding, :].reshape(data_y.shape)
#         pred_y = model(data_x).reshape(data_y.shape)
        if i <=100:
            test_x[i] = data_x.cpu().numpy()
            test_y[i] = data_y.cpu().numpy()
#         test_y0[i] = data_x[..., 0, -out_dim:].cpu().numpy() # same way as in training code
            preds_y[i] = pred_y.cpu().numpy()


for i in range(len(preds_y)):
    x_size,t_size=preds_y.shape[1],preds_y.shape[2]
    for m in range(x_size):
        for n in range(t_size):
            if preds_y[i,m,n,0]<0:
                preds_y[i,m,n,0]=0

key = 0
# # Save and Load Data
def save_data(data_path, test_x, test_y, preds_y):
    data_dir, data_filename = os.path.split(data_path)
    os.makedirs(data_dir, exist_ok=True)
    np.savez(data_path, test_x=test_x, test_y=test_y, preds_y=preds_y)

def load_data(data_path):
    data = np.load(data_path)
    test_x = data['test_x']
    test_y = data['test_y']
    preds_y = data['preds_y']
    return test_x, test_y, preds_y

data_dir = 'data/1d'
data_filename = 'data.npz'
data_path = os.path.join(data_dir, data_filename)
# os.makedirs(data_dir, exist_ok=True)
save_data(data_path, test_x[:100], test_y[:100], preds_y[:100])
test_x, test_y, preds_y = load_data(data_path)

os.makedirs(savefig_path, exist_ok=True)
#output plt
for count in range(len(preds_y[:10])):
    with open(savefig_path+ 'example_1d_{0:d}.dat'.format(count), 'w') as f:
        for t in range(preds_y.shape[2]):
            f.writelines('VARIABLES ="X","H","U","surface" ''\n')
            output_data = preds_y[count]
            output_z=dataset3.search_z(count).cpu().numpy()
            #output_z=test_x[count]
            clos = output_data.shape[0]
            f.writelines('Zone T="{}" i={},j={}'.format(t,clos,1) + '\n')
            for clo in range(clos):
                out_h = str(output_data[clo,t,0])
                out_u = str(output_data[clo,t,1])
                out_surface = str(output_data[clo, t, 0]+output_z[clo,t])
                f.writelines(str(xmin+dx*clo) + ' ' + out_h + ' ' + out_u +' '+out_surface+'\n')
    f.close()

def generate_movie_1D(key, test_x, test_y, preds_y, plot_title='', movie_dir='', movie_name='movie.gif',
                      frame_basename='movie', frame_ext='jpg', remove_frames=True, font_size=None,xmin=0,xmax=1,tend=0,z=0):
    frame_files = []
    os.makedirs(movie_dir, exist_ok=True)

    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})

    pred_h = preds_y[key,:,:,0]
    true_h = test_y[key,:,:,0]
    true_z=z
    #true_z=test_x[key,:,:,4]*1400

    a = test_x[key]
    Nx,Nt, _ = a.shape
    h0 = a[:, 0, -1]
    T = a[:, :, 1]*tend
    X = a[:, :, 0]*xmax
    x = X[:,0]
    t = T[0,:]

    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(111)

    xlim = [xmin, xmax]
    #plt.ylim()
    #     plt.xlabel(f'$x$')
    #     plt.ylabel(f'$u$')
    #     plt.title(f'{plot_title} $t={t[0]:.2f}$')
    #     plt.legend(loc='lower right')
    plt.tight_layout()

    for i in range(Nt):
        ax.clear()
        ax.plot(x, true_z[:, i] , 'grey', label='Terrian', linewidth=0.5)
        ax.plot(x, true_z[:,i]+true_h[:,i], 'b-', label='Numerical solution')
        ax.plot(x, true_z[:,i]+pred_h[:,i], 'r:', label='FNO')
        #plt.ylim(ylim)
        plt.xlim(xlim)
        plt.xlabel(f'Distance(m)')
        plt.ylabel(f'Elevation(m)')
        plt.title(f'{plot_title} $t={t[i]:.2f}$')
        plt.legend(loc='upper right')
        plt.tight_layout()
        fig.canvas.draw()
        #         plt.show()
        if movie_dir:
            frame_path = os.path.join(movie_dir, f'{frame_basename}-{i:03}.{frame_ext}')
            frame_files.append(frame_path)
            plt.savefig(frame_path)

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

# ### Movie $\eta$
data_dir = 'data/1d'
data_filename = 'data.npz'
data_path = os.path.join(data_dir, data_filename)
test_x, test_y, preds_y = load_data(data_path)
movie_dir = f'1d/movie/'
os.makedirs(movie_dir, exist_ok=True)

key = 0
frame_ext = 'jpg'
plot_title = "landslide_1d"
font_size = 12
remove_frames = True
for key in range(len(preds_y[:10])):
    movie_name = f'1d_{key}.gif'
    frame_basename = f'1d_{key}_frame'
    z=dataset3.search_z(key)
    generate_movie_1D(key, test_x, test_y, preds_y, plot_title=plot_title, movie_dir=movie_dir,
                  movie_name=movie_name, frame_basename=frame_basename, frame_ext=frame_ext,
                  remove_frames=remove_frames, font_size=font_size,xmin=xmin,xmax=xmax,tend=tend,z=z)
