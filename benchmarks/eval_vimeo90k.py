import os
import sys
import argparse
sys.path.append('.')
import torch
import numpy as np
from utils import read
from metric import calculate_dists, calculate_psnr, calculate_ssim, calculate_lpips
import config

parser = argparse.ArgumentParser(description='IFRNet Evaluation')
parser.add_argument('--model_name', default='IFRNet', type=str, help='IFRNet, IFRNet_L, IFRNet_S, IFRNet_RGB2')
parser.add_argument('--checkpoint_path', default=None, type=str, help='Укажите путь к .pth, если имя файла нестандартное')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model_name == 'IFRNet':
    from models.IFRNet import Model
elif args.model_name == 'IFRNet_L':
    from models.IFRNet_L import Model
elif args.model_name == 'IFRNet_S':
    from models.IFRNet_S import Model
elif args.model_name == 'IFRNet_RGB2':
    from models.IFRNET_RGB2 import Model


model = Model()


if args.checkpoint_path:
    ckpt_path = args.checkpoint_path
else:
    ckpt_path = f'checkpoint/{args.model_name}/{args.model_name}_latest.pth'
    
    # if args.model_name == 'IFRNet' and not os.path.exists(ckpt_path):
    #     ckpt_path = 'checkpoints/IFRNet/IFRNet_Vimeo90K.pth'

print(f"Загрузка весов из: {ckpt_path}")
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
model.cuda()

path = config.VIMEO_DIR + '/'
f = open(path + 'tri_testlist.txt', 'r')

psnr_list = []
ssim_list = []
lpips_list = []
dists_list = []

for i in f:
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    print(path + 'sequences/' + name + '/im1.png')
    I0 = read(path + 'sequences/' + name + '/im1.png')
    I1 = read(path + 'sequences/' + name + '/im2.png')
    I2 = read(path + 'sequences/' + name + '/im3.png')
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
    embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

    I1_pred = model.inference(I0, I2, embt)

    psnr = calculate_psnr(I1_pred, I1).detach().cpu().numpy()
    ssim = calculate_ssim(I1_pred, I1).detach().cpu().numpy()
    lpips_val = calculate_lpips(I1_pred, I1)
    dists_val = calculate_dists(I1_pred, I1)

    psnr_list.append(psnr)
    ssim_list.append(ssim)
    lpips_list.append(lpips_val)
    dists_list.append(dists_val)
    
    print('Avg PSNR: {:.3f} SSIM: {:.4f}, LPIPS: {:.4f}, DISTS: {:4f}'.format(
            np.mean(psnr_list), 
            np.mean(ssim_list), 
            np.mean(lpips_list),
            np.mean(dists_list)
        )
    )