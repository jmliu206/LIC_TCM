import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM
import warnings
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return -10 * math.log10(1-ms_ssim(a, b, data_range=1.).item())

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    p = 128
    path = args.data
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=128, M=320)
    net = net.to(device)
    net.eval()
    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    dictory = {}
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)
    if args.real:
        net.update()
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
            x = img.unsqueeze(0)
            x_padded, padding = pad(x, p)
            count += 1
            with torch.no_grad():
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_enc = net.compress(x_padded)
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)
                out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                num_pixels = x.size(0) * x.size(2) * x.size(3)
                print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
                print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
                print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
                Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                PSNR += compute_psnr(x, out_dec["x_hat"])
                MS_SSIM += compute_msssim(x, out_dec["x_hat"])

    else:
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).convert('RGB')
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            x_padded, padding = pad(x, p)
            count += 1
            with torch.no_grad():
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_net = net.forward(x_padded)
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)
                out_net['x_hat'].clamp_(0, 1)
                out_net["x_hat"] = crop(out_net["x_hat"], padding)
                print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
                print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.2f}dB')
                print(f'Bit-rate: {compute_bpp(out_net):.3f}bpp')
                PSNR += compute_psnr(x, out_net["x_hat"])
                MS_SSIM += compute_msssim(x, out_net["x_hat"])
                Bit_rate += compute_bpp(out_net)
    PSNR = PSNR / count
    MS_SSIM = MS_SSIM / count
    Bit_rate = Bit_rate / count
    total_time = total_time / count
    print(f'average_PSNR: {PSNR:.2f}dB')
    print(f'average_MS-SSIM: {MS_SSIM:.4f}')
    print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
    print(f'average_time: {total_time:.3f} ms')
    

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
    