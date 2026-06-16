import os
import sys
import argparse
sys.path.append('.')
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

parser = argparse.ArgumentParser(description='IFRNet Speed and Parameters Benchmark')
parser.add_argument('--model_name', default='IFRNet', type=str, help='IFRNet, IFRNet_L, IFRNet_S, IFRNet_RGB2')
args = parser.parse_args()

if args.model_name == 'IFRNet':
    from models.IFRNet import Model
elif args.model_name == 'IFRNet_L':
    from models.IFRNet_L import Model
elif args.model_name == 'IFRNet_S':
    from models.IFRNet_S import Model
elif args.model_name == 'IFRNet_RGB2':
    from models.IFRNET_RGB2 import Model

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

img0 = torch.randn(1, 3, 256, 448).cuda()
img1 = torch.randn(1, 3, 256, 448).cuda()
embt = torch.tensor(1/2).float().view(1, 1, 1, 1).cuda()

model = Model().cuda().eval()

print('Calculating FLOPs and Params...')

original_forward = model.forward
model.forward = model.inference

flops, params = profile(model, inputs=(img0, img1, embt))
flops, params = clever_format([flops, params], "%.3f")
print(f'FLOPs: {flops}, Params (by thop): {params}')

model.forward = original_forward

total = sum([param.nelement() for param in model.parameters()])
print('Parameters (manual count): {:.2f}M'.format(total / 1e6))

with torch.no_grad():
    # Прогрев
    for i in range(10):
        out = model.inference(img0, img1, embt) 
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_stamp = time.time()
    
    # Чистый замер
    for i in range(100):
        out = model.inference(img0, img1, embt)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print('Inference Time: {:.3f}s / frame'.format((time.time() - time_stamp) / 100))