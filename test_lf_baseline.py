import numpy as np
from utils import psnr
from skimage import io 
from cv2 import cvtColor, COLOR_RGB2YCrCb
from skimage.metrics import structural_similarity as ssim

lf_path = 'lf/mytest/origami.png'
lf = io.imread(lf_path)
lf_y = cvtColor(lf, COLOR_RGB2YCrCb)[:,:,0].copy()
u, v = 2, 2
lf_uv = lf_y[504*u:504*(u+1), 504*v:504*(v+1)]
lf_base = np.tile(lf_uv, (4, 4))

# lf_pred_path = 'lf/result/bedroom.png_LF_luminance_ratio_25_epoch_160_PSNR_3.39_SSIM_-1.0000.png'
# lf_pred = io.imread(lf_pred_path)

rst_psnr = psnr(lf_base, lf_y)
rst_ssim = ssim(lf_base, lf_y)
print('baseline')
print('psnr = {}'.format(rst_psnr))
print('ssim = {}'.format(rst_ssim))

# rst_psnr = psnr(lf_y, lf_pred)
# rst_ssim = ssim(lf_y, lf_pred)
# print('cs_ratio = 25')
# print('psnr = {}'.format(rst_psnr))
# print('ssim = {}'.format(rst_ssim))