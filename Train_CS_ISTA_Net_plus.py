import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
from time import time
from ista_net_plus import ISTANetplus
from logger import Logger

parser = ArgumentParser(description='ISTA-Net-plus')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=4, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--data_name', type=str, default='Training_Data.mat', help='training data name')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--var_name', type=str, default='labels', help='var name of data')

args = parser.parse_args()


start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list

# redirect output to file
log_file_name = "./%s/Log_CS_ISTA_Net_plus_layer_%d_group_%d_ratio_%d_lr_%.4f.txt" % (
    args.log_dir, layer_num, group_num, cs_ratio, learning_rate)
logger = Logger(log_file_name)
sys.stdout = logger
logger.start()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Using device {}'.format(device))

ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089
nrtrain = 88912   # number of training blocks
batch_size = 64


# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']


Training_data_Name = args.data_name
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data[args.var_name]


Qinit_Name = './%s/Initialization_Matrix_%d.mat' % (args.matrix_dir, cs_ratio)


# Computing Initialization Matrix:
if os.path.exists(Qinit_Name):
    Qinit_data = sio.loadmat(Qinit_Name)
    Qinit = Qinit_data['Qinit']

else:
    X_data = Training_labels.transpose()
    Y_data = np.dot(Phi_input, X_data)
    Y_YT = np.dot(Y_data, Y_data.transpose())
    X_YT = np.dot(X_data, Y_data.transpose())
    Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
    del X_data, Y_data, X_YT, Y_YT
    sio.savemat(Qinit_Name, {'Qinit': Qinit})


model = ISTANetplus(layer_num)
model = nn.DataParallel(model)
model = model.to(device)



print_flag = 1   # print parameter number

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())



class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/CS_ISTA_Net_plus_layer_%d_group_%d_ratio_%d_lr_%.4f" % (args.model_dir, layer_num, group_num, cs_ratio, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))


Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor).to(device)
Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor).to(device)

logger.flush()

# Training loop
for epoch_i in range(start_epoch+1, end_epoch+1):

    epoch_start_time = time()
    cnt = 0
    for data in rand_loader:

        batch_x = data
        batch_x = batch_x.to(device)

        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))
        print('batch_x.shape = {}, Phi.shape = {}, Phix.shape = {}'.format(batch_x.shape, Phi.shape, Phix.shape))

        [x_output, loss_layers_sym] = model(Phix, Phi, Qinit)

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(layer_num-1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

        gamma = torch.Tensor([0.01]).to(device)

        # loss_all = loss_discrepancy
        loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        logger.stop()
        output_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f,  Constraint Loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item(), loss_constraint)
        print(output_data)

        cnt += 1
        # if cnt == 5:
        #     break

    logger.start()
    logger.flush()
    epoch_end_time = time()
    print('Epoch %d takes %.2f secs' % (epoch_i, epoch_end_time - epoch_start_time))
    print(output_data)

    if epoch_i % 5 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
