import torch
import torch.nn as nn
import scipy.io as sio
import os
import glob
from time import time
import cv2
from skimage.metrics import structural_similarity as ssim
from argparse import ArgumentParser
from utils import *
from ista_net_plus import ISTANetplus

################
##  Argparse  ##
################

parser = ArgumentParser(description='ISTA-Net-plus')

parser.add_argument('--epoch_num', type=int, default=160, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=30, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix',
                    help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model',
                    help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')

args = parser.parse_args()

epoch_num = args.epoch_num
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name

#############
##  Setup  ##
#############

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089
nrtrain = 88912
batch_size = 64

# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']

Qinit_Name = './%s/Initialization_Matrix_%d.mat' % (args.matrix_dir, cs_ratio)

# Computing Initialization Matrix:
if os.path.exists(Qinit_Name):
    Qinit_data = sio.loadmat(Qinit_Name)
    Qinit = Qinit_data['Qinit']

else:
    Training_data_Name = 'Training_Data.mat'
    Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
    Training_labels = Training_data['labels']

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

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/CS_ISTA_Net_plus_layer_%d_group_%d_ratio_%d_lr_%.4f" % (
    args.model_dir, layer_num, group_num, cs_ratio, learning_rate)

# Load pre-trained model with epoch number
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num),
                                 map_location=device))



test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*')

result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)

Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
Qinit = Qinit.to(device)


######################
##  Reconstruction  ##
######################

print('\n')
print("CS Reconstruction Start")

with torch.no_grad():
    for img_no in range(ImgNum):
        imgName = filepaths[img_no]

        Img = cv2.imread(imgName, 1)

        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()

        Iorg_y = Img_yuv[:, :, 0]

        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
        Icol = img2col_py(Ipad, 33).transpose() / 255.0

        Img_output = Icol

        start = time()

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)

        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

        [x_output, loss_layers_sym] = model(Phix, Phi, Qinit)

        end = time()

        Prediction_value = x_output.cpu().data.numpy()

        # loss_sym = torch.mean(torch.pow(loss_layers_sym[0], 2))
        # for k in range(layer_num - 1):
        #     loss_sym += torch.mean(torch.pow(loss_layers_sym[k + 1], 2))
        #
        # loss_sym = loss_sym.cpu().data.numpy()

        X_rec = np.clip(col2im_CS_py(Prediction_value.transpose(), row, col, row_new, col_new), 0,
                        1)

        rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)

        print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (
            img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

        Img_rec_yuv[:, :, 0] = X_rec * 255

        im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

        resultName = imgName.replace(args.data_dir, args.result_dir)
        cv2.imwrite("%s_ISTA_Net_plus_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.png" % (
            resultName, cs_ratio, epoch_num, rec_PSNR, rec_SSIM), im_rec_rgb)
        del x_output

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

print('\n')
output_data = "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n" % (
    cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
print(output_data)

output_file_name = "./%s/PSNR_SSIM_Results_CS_ISTA_Net_plus_layer_%d_group_%d_ratio_%d_lr_%.4f.txt" % (
    args.log_dir, layer_num, group_num, cs_ratio, learning_rate)

output_file = open(output_file_name, 'a')
output_file.write(output_data)
output_file.close()

print("CS Reconstruction End")
